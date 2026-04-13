from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path("notebooks/kernel_mechanism_walkthrough.ipynb")


def _source(text: str) -> list[str]:
    return dedent(text).lstrip("\n").splitlines(keepends=True)


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def build_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # Kernel Mechanism Walkthrough

            This notebook starts **exactly where** [inverse_problem_walkthrough.ipynb](./inverse_problem_walkthrough.ipynb) ends.

            That notebook finishes by producing three toy slice graphs and their WL feature matrix:

            - `shared_signal`
            - `same_circuit`
            - `other_circuit`

            Here we reuse that same toy setup and answer the next question:

            > Once we already have a WL matrix, what do the classical kernels actually do with it?

            The notebook focuses only on the kernel stage:

            1. rebuild the same toy WL matrix from the inverse-problem notebook,
            2. verify the bridge with the old cosine-similarity view,
            3. show how the **linear** and **RBF** kernels turn WL vectors into similarities,
            4. and make the classifier geometry visible using small perturbations around those same WL rows.
            """
        ),
        markdown_cell(
            """
            ## Bridge between the two notebooks

            The important relationship is:

            - `inverse_problem_walkthrough.ipynb` explains how we get from an observed patch-effect matrix to a graph and then to a WL matrix.
            - this notebook assumes that WL matrix is already available and explains what the **kernel layer** is doing next.

            So this notebook is not a different toy example. It is the **continuation** of the same toy example.
            """
        ),
        code_cell(
            r"""
            import numpy as np
            import matplotlib.pyplot as plt

            from sklearn.decomposition import PCA
            from sklearn.metrics.pairwise import cosine_similarity, linear_kernel, rbf_kernel
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC

            from pig.embeddings import compute_wl_features
            from pig.graph import GraphBuilder
            from pig.kernels import ClassicalKernelClassifier
            from pig.patching import ComponentSpec, PatchEffectDataset, PatchEffectTensor
            from pig.prompts import PromptPair, SliceLabel

            plt.style.use("seaborn-v0_8-whitegrid")
            plt.rcParams.update(
                {
                    "figure.dpi": 140,
                    "savefig.dpi": 200,
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                    "axes.titleweight": "bold",
                    "axes.titlesize": 13,
                    "axes.labelsize": 11,
                    "legend.frameon": False,
                }
            )

            rng = np.random.default_rng(7)
            """
        ),
        markdown_cell(
            """
            ## Stage 1. Rebuild the exact toy WL matrix from the inverse-problem notebook

            These are the same toy effect matrices used at the end of `inverse_problem_walkthrough.ipynb`.

            We are not re-explaining why they exist; we are only reconstructing the same endpoint so the two notebooks line up exactly.
            """
        ),
        markdown_cell(
            """
            ### How to read the WL matrix

            Before looking at the heatmap, here is what each part means:

            - **Rows**: each row is one toy slice graph:
              - `shared_signal`
              - `same_circuit`
              - `other_circuit`
            - **Columns**: each column is one **active WL feature**.
              In the notebook they appear as `f9`, `f21`, `f22`, etc.
              These are short column ids taken from the full WL vocabulary.
            - **Cell value**: the number inside each cell is the **count** of that WL feature in that slice graph.
              In this toy example the counts are mostly `0` or `1`.
            - **Color**: darker blue means a larger WL feature count.
            - **Color bar**: maps the heatmap color back to the numeric WL count.

            So the whole matrix should be read as:

            > each row is one graph, each column is one structural WL pattern, and each cell says how strongly that pattern appears in that graph.

            That is exactly the object the kernel stage consumes next.
            """
        ),
        code_cell(
            r"""
            node_labels = ["L0T0", "L0T1", "L1T0", "L1T1"]
            component_axis = [ComponentSpec(node_type="res")]

            def make_dataset(matrix, slice_label):
                ds = PatchEffectDataset()
                for i in range(matrix.shape[0]):
                    pair = PromptPair(
                        x_cln=f"clean-{slice_label.corruption}-{i}",
                        x_crp=f"corrupt-{slice_label.corruption}-{i}",
                        y_star="target",
                        slice_label=slice_label,
                        meta={},
                    )
                    effects = matrix[i].reshape(2, 2, 1).astype(np.float32)
                    ds.add(
                        PatchEffectTensor(
                            effects=effects,
                            component_axis=component_axis,
                            prompt_pair=pair,
                            base_score=-1.0,
                            clean_score=1.0,
                        )
                    )
                return ds

            X_obs = np.array([
                [ 2.0,  1.8, -1.6,  0.1],
                [ 1.0,  0.9, -0.8, -0.1],
                [ 0.0,  0.1,  0.0,  0.2],
                [-1.0, -0.8,  0.7,  0.0],
                [-2.0, -1.7,  1.5, -0.2],
                [-1.0, -0.9,  0.8,  0.1],
            ], dtype=np.float32)

            X_same_circuit = np.array([
                [ 1.8,  1.6, -1.5,  0.1],
                [ 0.9,  0.8, -0.8,  0.0],
                [ 0.1,  0.0, -0.1,  0.1],
                [-0.9, -0.8,  0.7,  0.0],
                [-1.8, -1.6,  1.4, -0.1],
                [-0.9, -0.8,  0.7,  0.1],
            ], dtype=np.float32)

            X_other_circuit = np.array([
                [ 1.8,  0.1, -0.2,  1.6],
                [ 0.9,  0.0, -0.1,  0.8],
                [ 0.0,  0.1,  0.0,  0.1],
                [-0.9,  0.1,  0.2, -0.8],
                [-1.8, -0.1,  0.2, -1.6],
                [-0.9,  0.0,  0.1, -0.8],
            ], dtype=np.float32)

            slice_obs = SliceLabel(task="inverse_demo", corruption="shared_signal")
            slice_same = SliceLabel(task="inverse_demo", corruption="same_circuit")
            slice_other = SliceLabel(task="inverse_demo", corruption="other_circuit")

            builder = GraphBuilder(k=2, enforce_direction=True)
            graph_obs = builder.build_from_slice(make_dataset(X_obs, slice_obs), slice_obs)
            graph_same = builder.build_from_slice(make_dataset(X_same_circuit, slice_same), slice_same)
            graph_other = builder.build_from_slice(make_dataset(X_other_circuit, slice_other), slice_other)

            graphs = {
                slice_obs: graph_obs,
                slice_same: graph_same,
                slice_other: graph_other,
            }
            feature_matrix = compute_wl_features(graphs, depth=2)
            X_wl = feature_matrix.to_matrix()
            slice_names = [s.corruption for s in feature_matrix.slice_labels]

            feature_var = X_wl.var(axis=0)
            active_idx = [idx for idx in np.argsort(feature_var)[::-1] if feature_var[idx] > 0][:12]
            if not active_idx:
                active_idx = list(range(min(12, X_wl.shape[1])))
            X_wl_view = X_wl[:, active_idx]
            feature_labels = [f"f{idx}" for idx in active_idx]

            print("WL matrix shape:", X_wl.shape)
            print("Slice order:", slice_names)

            fig, ax = plt.subplots(figsize=(9.5, 3.8))
            im = ax.imshow(X_wl_view, aspect="auto", cmap="Blues")
            ax.set_xticks(np.arange(len(feature_labels)))
            ax.set_xticklabels(feature_labels, rotation=35, ha="right")
            ax.set_yticks(np.arange(len(slice_names)))
            ax.set_yticklabels(slice_names)
            ax.set_title("Same toy WL matrix we reached in the inverse-problem notebook")
            for i in range(X_wl_view.shape[0]):
                for j in range(X_wl_view.shape[1]):
                    ax.text(j, i, f"{int(X_wl_view[i, j])}", ha="center", va="center", fontsize=9)
            plt.colorbar(im, ax=ax, label="WL feature count")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## Stage 2. Verify the bridge with the old similarity view

            In the previous notebook, the final view was a **cosine similarity** over WL features.

            We start by reproducing that same lens, so the handoff between notebooks is explicit.
            """
        ),
        code_cell(
            r"""
            K_cosine = cosine_similarity(X_wl)

            fig, ax = plt.subplots(figsize=(5.8, 4.8))
            im = ax.imshow(K_cosine, cmap="magma", vmin=0.0, vmax=1.0)
            ax.set_xticks(np.arange(len(slice_names)))
            ax.set_xticklabels(slice_names, rotation=35, ha="right")
            ax.set_yticks(np.arange(len(slice_names)))
            ax.set_yticklabels(slice_names)
            ax.set_title("Cosine similarity over the same toy WL matrix")
            for i in range(K_cosine.shape[0]):
                for j in range(K_cosine.shape[1]):
                    ax.text(j, i, f"{K_cosine[i, j]:.2f}", ha="center", va="center", fontsize=10, color="white" if K_cosine[i, j] < 0.55 else "black")
            plt.colorbar(im, ax=ax, label="cosine similarity")
            plt.tight_layout()
            plt.show()

            print("Pairwise cosine similarities")
            for i, a in enumerate(slice_names):
                for j, b in enumerate(slice_names):
                    print(f"  {a:>13} vs {b:<13}: {K_cosine[i, j]:.3f}")
            """
        ),
        markdown_cell(
            """
            ## Stage 3. What changes when we move from cosine to the actual classical kernels

            The repo's classical kernel path does not use cosine similarity as the classifier itself.

            It uses:

            - `StandardScaler`
            - `SVC(kernel="linear")` or `SVC(kernel="rbf")`

            So the next step is to take this same WL matrix and look at the similarities induced by those kernels.
            """
        ),
        markdown_cell(
            """
            ### Why do we apply `StandardScaler`?

            This is one of the most important design choices in the real pipeline.

            The reason is geometric:

            - the **linear kernel** uses dot products,
            - the **RBF kernel** uses Euclidean distances,
            - and both are sensitive to feature scale.

            If one WL feature has a much larger numeric range than the others, it can dominate the similarity computation even if it is not the most informative feature structurally.

            So in the real pipeline we standardize feature-by-feature:

            $$
            x'_{ij} = \\frac{x_{ij} - \\mu_j}{\\sigma_j}
            $$

            where:

            - $x_{ij}$ is the original value of feature $j$ in slice $i$,
            - $\\mu_j$ is the mean of feature $j$,
            - $\\sigma_j$ is the standard deviation of feature $j$.

            In other words, each column is re-centered and re-scaled so that no single WL feature wins just because of its raw numeric scale.
            """
        ),
        code_cell(
            r"""
            scaler = StandardScaler()
            X_wl_scaled = scaler.fit_transform(X_wl)

            scaled_view = X_wl_scaled[:, active_idx]

            fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
            im0 = axes[0].imshow(X_wl_view, aspect="auto", cmap="Blues")
            axes[0].set_title("Raw active WL features")
            axes[0].set_xticks(np.arange(len(feature_labels)))
            axes[0].set_xticklabels(feature_labels, rotation=35, ha="right")
            axes[0].set_yticks(np.arange(len(slice_names)))
            axes[0].set_yticklabels(slice_names)
            for i in range(X_wl_view.shape[0]):
                for j in range(X_wl_view.shape[1]):
                    axes[0].text(j, i, f"{int(X_wl_view[i, j])}", ha="center", va="center", fontsize=9)
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            vmax = float(np.max(np.abs(scaled_view)))
            im1 = axes[1].imshow(scaled_view, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[1].set_title("Same features after StandardScaler")
            axes[1].set_xticks(np.arange(len(feature_labels)))
            axes[1].set_xticklabels(feature_labels, rotation=35, ha="right")
            axes[1].set_yticks(np.arange(len(slice_names)))
            axes[1].set_yticklabels(slice_names)
            for i in range(scaled_view.shape[0]):
                for j in range(scaled_view.shape[1]):
                    axes[1].text(j, i, f"{scaled_view[i, j]:+.1f}", ha="center", va="center", fontsize=8)
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell(
            """
            ### What can we see in the scaling figure?

            - The **left panel** is the actual WL feature block the kernel receives.
            - The **right panel** is the same block after standardization.

            What changes:

            - raw counts like `0` and `1` become positive or negative z-scores,
            - a positive number now means “this slice has more of this feature than average,”
            - a negative number means “this slice has less of this feature than average.”

            So after scaling, the kernel is no longer comparing raw counts directly.
            It is comparing each slice in terms of **relative over-expression or under-expression of structural patterns**.
            """
        ),
        markdown_cell(
            """
            ## Stage 4. Linear kernel on the inverse-problem WL matrix

            The linear kernel is just a dot product between scaled WL rows:

            $$
            K_{\\text{linear}}(x_i, x_j) = x_i^\\top x_j
            $$

            So it measures whether two slices point in a similar direction in WL feature space.
            """
        ),
        code_cell(
            r"""
            K_linear = linear_kernel(X_wl_scaled, X_wl_scaled)

            fig, ax = plt.subplots(figsize=(5.8, 4.8))
            im = ax.imshow(K_linear, cmap="RdBu_r")
            ax.set_xticks(np.arange(len(slice_names)))
            ax.set_xticklabels(slice_names, rotation=35, ha="right")
            ax.set_yticks(np.arange(len(slice_names)))
            ax.set_yticklabels(slice_names)
            ax.set_title("Linear kernel similarity on the same WL matrix")
            for i in range(K_linear.shape[0]):
                for j in range(K_linear.shape[1]):
                    ax.text(j, i, f"{K_linear[i, j]:+.1f}", ha="center", va="center", fontsize=10, color="white" if abs(K_linear[i, j]) > 7 else "black")
            plt.colorbar(im, ax=ax, label="linear-kernel similarity")
            plt.tight_layout()
            plt.show()

            print("Linear-kernel similarities")
            for i, a in enumerate(slice_names):
                for j, b in enumerate(slice_names):
                    print(f"  {a:>13} vs {b:<13}: {K_linear[i, j]:+.3f}")
            """
        ),
        markdown_cell(
            """
            ### How to read the linear-kernel heatmap

            In this matrix:

            - a **large positive** value means two slices activate similar standardized WL patterns,
            - a **negative** value means their WL patterns point in opposite directions,
            - the diagonal is large because each slice is perfectly similar to itself.

            So the main question is not whether values are big in absolute terms, but whether the **relative structure** matches the story we expect.

            Here you can already see that `shared_signal` and `same_circuit` are closer to each other than either is to `other_circuit`.
            """
        ),
        markdown_cell(
            """
            ## Stage 5. RBF kernel on the same WL matrix

            The RBF kernel uses distance:

            $$
            K_{\\text{RBF}}(x_i, x_j) = \\exp\\left(-\\gamma \\lVert x_i - x_j \\rVert^2\\right)
            $$

            This makes the notion of similarity more local. Two slices only look similar if their WL rows are truly close.
            """
        ),
        code_cell(
            r"""
            gamma = 0.02
            K_rbf = rbf_kernel(X_wl_scaled, X_wl_scaled, gamma=gamma)

            fig, ax = plt.subplots(figsize=(5.8, 4.8))
            im = ax.imshow(K_rbf, cmap="magma", vmin=0.0, vmax=1.0)
            ax.set_xticks(np.arange(len(slice_names)))
            ax.set_xticklabels(slice_names, rotation=35, ha="right")
            ax.set_yticks(np.arange(len(slice_names)))
            ax.set_yticklabels(slice_names)
            ax.set_title(f"RBF kernel similarity on the same WL matrix (gamma={gamma})")
            for i in range(K_rbf.shape[0]):
                for j in range(K_rbf.shape[1]):
                    ax.text(j, i, f"{K_rbf[i, j]:.2f}", ha="center", va="center", fontsize=10, color="white" if K_rbf[i, j] < 0.55 else "black")
            plt.colorbar(im, ax=ax, label="RBF similarity")
            plt.tight_layout()
            plt.show()

            print("RBF-kernel similarities")
            for i, a in enumerate(slice_names):
                for j, b in enumerate(slice_names):
                    print(f"  {a:>13} vs {b:<13}: {K_rbf[i, j]:.3f}")
            """
        ),
        markdown_cell(
            """
            ### How to read the RBF heatmap

            The RBF kernel turns distances into similarities between `0` and `1`.

            That means:

            - `1.00` means “identical to itself,”
            - values closer to `1` mean “very close in WL space,”
            - values closer to `0` mean “far apart.”

            Compared with the linear kernel, the RBF kernel gives a more **local** notion of similarity.
            It is less about overall direction and more about neighborhood closeness.
            """
        ),
        markdown_cell(
            """
            ## Stage 6. The same three slices, now seen through three similarity lenses

            This is the cleanest way to relate the notebooks:

            - the inverse-problem notebook ends with the **cosine** view,
            - this notebook shows what the **linear** and **RBF** kernel views do to the same rows.
            """
        ),
        code_cell(
            r"""
            fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
            for ax, K, title, cmap, vmin, vmax in [
                (axes[0], K_cosine, "Cosine", "magma", 0.0, 1.0),
                (axes[1], K_linear, "Linear kernel", "RdBu_r", None, None),
                (axes[2], K_rbf, "RBF kernel", "magma", 0.0, 1.0),
            ]:
                im = ax.imshow(K, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_xticks(np.arange(len(slice_names)))
                ax.set_xticklabels(slice_names, rotation=35, ha="right")
                ax.set_yticks(np.arange(len(slice_names)))
                ax.set_yticklabels(slice_names)
                ax.set_title(title)
                for i in range(K.shape[0]):
                    for j in range(K.shape[1]):
                        label = f"{K[i, j]:+.1f}" if title == "Linear kernel" else f"{K[i, j]:.2f}"
                        color = "white" if (title == "Linear kernel" and abs(K[i, j]) > 7) or (title != "Linear kernel" and K[i, j] < 0.55) else "black"
                        ax.text(j, i, label, ha="center", va="center", fontsize=9, color=color)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.suptitle("Three similarity views over the exact same toy WL matrix", y=1.02)
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell(
            """
            ### Reality check: this comparison is the real kernel story

            Up to this point, everything is already faithful to the real classical pipeline:

            1. start from a WL matrix,
            2. scale it,
            3. compute similarities induced by the kernel.

            This is the core reality.

            The next section with PCA is **not** the real classifier space.
            It is only a 2D picture that helps us look at a decision boundary with human eyes.
            """
        ),
        markdown_cell(
            """
            ## Stage 7. What the real classifier actually does

            In the actual pipeline, we do **not** project to 2D first.

            The real flow is:

            - take the full WL matrix,
            - fit a `StandardScaler`,
            - fit an `SVC(kernel="linear")` or `SVC(kernel="rbf")`,
            - evaluate with cross-validation.

            So before any PCA picture, we expose the actual estimator object used by the repo.
            """
        ),
        code_cell(
            r"""
            real_linear_clf = ClassicalKernelClassifier(kernel="linear", random_state=42, normalize=True)
            real_rbf_clf = ClassicalKernelClassifier(kernel="rbf", random_state=42, normalize=True)

            print("Actual estimator used for CV (linear):")
            print(real_linear_clf._create_cv_estimator())
            print()
            print("Actual estimator used for CV (rbf):")
            print(real_rbf_clf._create_cv_estimator())
            """
        ),
        markdown_cell(
            """
            ## Stage 8. Make the classifier geometry visible

            With only three slices, the kernel matrices are easy to read, but they are not great for visualizing a decision boundary.

            So here we do something purely pedagogical:

            - keep the exact three WL rows from the inverse-problem notebook as **anchors**,
            - generate small perturbation copies around them,
            - and define two families:
              - **reuse-family**: perturbations around `shared_signal` and `same_circuit`
              - **other-family**: perturbations around `other_circuit`

            That lets us see the geometry that the kernels are exploiting, while still staying anchored to the exact toy WL rows from the previous notebook.
            """
        ),
        code_cell(
            r"""
            anchor_lookup = {name: X_wl[idx] for idx, name in enumerate(slice_names)}

            augment_specs = [
                ("shared_signal", 0, 4, 0.22),
                ("same_circuit", 0, 4, 0.22),
                ("other_circuit", 1, 5, 0.22),
            ]

            X_aug = []
            y_aug = []
            point_names = []
            for anchor_name, label, copies, noise_scale in augment_specs:
                anchor = anchor_lookup[anchor_name]
                for copy_idx in range(copies):
                    noise = rng.normal(scale=noise_scale, size=anchor.shape)
                    X_aug.append(anchor + noise)
                    y_aug.append(label)
                    point_names.append(f"{anchor_name[:2]}-{copy_idx+1}")

            X_aug = np.asarray(X_aug, dtype=np.float32)
            y_aug = np.asarray(y_aug, dtype=np.int32)
            class_names = {0: "reuse-family", 1: "other-family"}

            print("Augmented WL matrix shape:", X_aug.shape)
            print("Class balance:", {class_names[int(k)]: int((y_aug == k).sum()) for k in np.unique(y_aug)})
            """
        ),
        markdown_cell(
            """
            ## Stage 9. Linear vs RBF decision geometry

            We project the augmented WL cloud to 2D with PCA.
            This is just for visualization; the real classifier still works in full WL space.

            So the rule is:

            - trust the kernel matrices and the actual estimator as the faithful part,
            - treat the PCA panels as an intuition aid.
            """
        ),
        code_cell(
            r"""
            scaler_aug = StandardScaler()
            X_aug_scaled = scaler_aug.fit_transform(X_aug)
            X_anchor_scaled = scaler_aug.transform(X_wl)

            pca = PCA(n_components=2, random_state=0)
            X_2d = pca.fit_transform(X_aug_scaled)
            X_anchor_2d = pca.transform(X_anchor_scaled)

            linear_vis = SVC(kernel="linear", C=1.0)
            rbf_vis = SVC(kernel="rbf", C=1.0, gamma=0.8)
            linear_vis.fit(X_2d, y_aug)
            rbf_vis.fit(X_2d, y_aug)

            def plot_decision_boundary(ax, model, title):
                x_min, x_max = X_2d[:, 0].min() - 1.0, X_2d[:, 0].max() + 1.0
                y_min, y_max = X_2d[:, 1].min() - 1.0, X_2d[:, 1].max() + 1.0
                xx, yy = np.meshgrid(
                    np.linspace(x_min, x_max, 320),
                    np.linspace(y_min, y_max, 320),
                )
                grid = np.c_[xx.ravel(), yy.ravel()]
                zz = model.decision_function(grid).reshape(xx.shape)

                ax.contourf(xx, yy, zz > 0, alpha=0.16, levels=1, colors=["#8ecae6", "#f4a261"])
                ax.contour(xx, yy, zz, levels=[0], colors="black", linewidths=1.5)

                colors = np.where(y_aug == 0, "#1d3557", "#d62828")
                for i in range(len(X_2d)):
                    ax.scatter(X_2d[i, 0], X_2d[i, 1], s=70, color=colors[i], edgecolor="white", linewidth=0.9)

                anchor_colors = ["#0f766e", "#0f766e", "#b91c1c"]
                for i, name in enumerate(slice_names):
                    ax.scatter(
                        X_anchor_2d[i, 0],
                        X_anchor_2d[i, 1],
                        s=180,
                        marker="X",
                        color=anchor_colors[i],
                        edgecolor="white",
                        linewidth=1.2,
                        zorder=5,
                    )
                    ax.text(X_anchor_2d[i, 0] + 0.08, X_anchor_2d[i, 1] + 0.08, name, fontsize=9, fontweight="bold")

                ax.set_title(title)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")

            fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
            plot_decision_boundary(axes[0], linear_vis, "Linear SVM over the anchored WL cloud")
            plot_decision_boundary(axes[1], rbf_vis, "RBF SVM over the anchored WL cloud")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## Stage 10. Use the same classifier object as the repo

            Finally, we run the actual `ClassicalKernelClassifier` object on the augmented WL cloud.

            That closes the loop:

            - the **anchor rows** come from the inverse-problem notebook,
            - the **geometry** comes from small perturbations around those rows,
            - and the **classifier** is the same object used by the repo's classical baseline.
            """
        ),
        code_cell(
            r"""
            linear_clf = ClassicalKernelClassifier(kernel="linear", random_state=42, normalize=True)
            rbf_clf = ClassicalKernelClassifier(kernel="rbf", random_state=42, normalize=True)

            linear_cv = linear_clf.cross_validate(X_aug, y_aug, cv=4)
            rbf_cv = rbf_clf.cross_validate(X_aug, y_aug, cv=4)

            print("Linear CV:", linear_cv)
            print("RBF CV:", rbf_cv)
            """
        ),
        markdown_cell(
            """
            ### What should we conclude from the CV result?

            The exact score here is not the main point, because the setup is still toy.

            What matters is that:

            - the same classifier object used by the repo can separate these anchored WL patterns,
            - both kernels operate directly on full WL rows,
            - and the notebook has shown both the **real mechanism** and a **visual aid** for intuition.
            """
        ),
        markdown_cell(
            """
            ## What this notebook demonstrates

            Relative to `inverse_problem_walkthrough.ipynb`, the message is now:

            1. the previous notebook gave us a toy WL matrix over slices,
            2. this notebook reuses that exact WL matrix,
            3. cosine, linear-kernel, and RBF-kernel views are three different similarity lenses over the same rows,
            4. and the classical kernel stage works by turning those structural similarities into a decision rule.

            So the kernel stage is not a separate mystery.
            It is simply the **next layer of comparison** built on top of the toy WL representation from the inverse-problem notebook.
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(), indent=1))
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
