import { GraphFilter, ViewerMessage } from "../types";

type MessageHandler = (message: ViewerMessage) => void;
type StatusHandler = (status: string) => void;

export class ViewerClient {
  private readonly socket: WebSocket;
  private readonly url: string;

  constructor(url: string, onMessage: MessageHandler, onStatus?: StatusHandler) {
    this.url = url;
    console.log(`[viewer][ws] connect ${this.url}`);
    this.socket = new WebSocket(url);
    this.socket.onopen = () => {
      console.log(`[viewer][ws] open ${this.url}`);
      onStatus?.("socket_open");
    };
    this.socket.onerror = (event) => {
      console.error(`[viewer][ws] error ${this.url}`, event);
      onStatus?.("socket_error");
    };
    this.socket.onclose = (event) => {
      console.warn(
        `[viewer][ws] close ${this.url} code=${event.code} reason="${event.reason}" wasClean=${event.wasClean}`,
      );
      onStatus?.(`socket_closed(${event.code})`);
    };
    this.socket.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data) as ViewerMessage;
        console.log(`[viewer][ws] message ${this.url} type=${parsed.type}`);
        onMessage(parsed);
      } catch {
        console.error(`[viewer][ws] invalid_message ${this.url}`, event.data);
        onStatus?.("invalid_message");
      }
    };
  }

  isOpen(): boolean {
    return this.socket.readyState === WebSocket.OPEN;
  }

  close(): void {
    console.log(`[viewer][ws] close() requested ${this.url}`);
    this.socket.close();
  }

  requestUpdate(filter: GraphFilter): void {
    if (!this.isOpen()) {
      console.log(
        `[viewer][ws] skip filter_update (socket not open) ${this.url} state=${this.socket.readyState}`,
      );
      return;
    }
    console.log(
      `[viewer][ws] send filter_update ${this.url} slice=${filter.slice_id} max_edges=${filter.max_edges} min_abs_weight=${filter.min_abs_weight}`,
    );
    this.socket.send(
      JSON.stringify({
        type: "filter_update",
        ...filter,
      }),
    );
  }
}
