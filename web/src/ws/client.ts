import { GraphFilter, ViewerMessage } from "../types";

type MessageHandler = (message: ViewerMessage) => void;

export class ViewerClient {
  private readonly socket: WebSocket;

  constructor(url: string, onMessage: MessageHandler) {
    this.socket = new WebSocket(url);
    this.socket.onmessage = (event) => {
      const parsed = JSON.parse(event.data) as ViewerMessage;
      onMessage(parsed);
    };
  }

  isOpen(): boolean {
    return this.socket.readyState === WebSocket.OPEN;
  }

  requestUpdate(filter: GraphFilter): void {
    if (!this.isOpen()) {
      return;
    }
    this.socket.send(
      JSON.stringify({
        type: "filter_update",
        ...filter,
      }),
    );
  }
}
