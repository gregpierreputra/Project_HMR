import gradio as gr

from demo_cli import HMRDemo, _parse_cli_args


def _main():
    args = _parse_cli_args(with_image_path=False)

    inputs = [
        gr.Image(
            type="numpy",
            label="Input image",
            value="demo/input_images/Feature_02_ActionPoses_mobile.jpg",
        )
    ]

    outputs = [
        gr.Image(label="Output all mesh"),
        gr.Model3D(label="3D Mesh"),
    ]

    hmr_demo = HMRDemo(args=args)

    interface = gr.Interface(hmr_demo.run, inputs, outputs)
    interface.launch()


if __name__ == "__main__":
    _main()
