"""Interactive prompts for configuration."""

from rich.console import Console
import questionary
from questionary import Style

from ..config import TEST_DIR, find_camera_card


custom_style = Style([("highlighted", "fg:#00ff00 bold")])


def _ask(question):
    """
    Run a questionary prompt, returning None when there is no one to answer.

    questionary raises EOFError when stdin is not a terminal (a pipe, cron, a
    redirect). Callers already handle None as "nothing selected", so map it
    onto that instead of crashing with a traceback.
    """
    try:
        return question.ask()
    except EOFError:
        return None


def default_args_prompt(args, console: Console):
    """Prompt for default settings vs custom configuration."""
    choices = [
        questionary.Choice(
            title="Default - (from memory card, With compression, use yolo11x_custom.pt)", 
            value=True
        ),
        questionary.Choice(
            title="With options - (Change input folder, compression and model)", 
            value=False
        ),
    ]

    result = _ask(questionary.select(
        "Run with default settings?",
        choices=choices,
        style=custom_style
    ))

    if result is None:
        console.print("No choice selected. Exiting.")
        exit(1)

    if result:
        card_dir = find_camera_card()
        if card_dir is None:
            console.print(
                f"[yellow]No camera card mounted. Using test folder:[/yellow] {TEST_DIR}"
            )
        default_input = str(card_dir) if card_dir else str(TEST_DIR)

        # Do not clobber values already given on the command line.
        if args.compress_files is None:
            args.compress_files = True
        if args.input_dir is None:
            args.input_dir = default_input
        if args.model is None:
            args.model = "yolo11x_custom.pt"


def interactive_compression_prompt(args, console: Console):
    """Prompt for compression settings."""
    if not hasattr(args, "compress_files") or args.compress_files is None:
        choices = [
            questionary.Choice(title="Yes - (saves space, slower)", value=True),
            questionary.Choice(title="No - (faster, more space)", value=False),
        ]

        result = _ask(questionary.select(
            "Enable file compression?",
            choices=choices,
            style=custom_style
        ))

        if result is not None:
            args.compress_files = result
        else:
            console.print("No compression option selected. Using default (enabled).")
            args.compress_files = True


def interactive_input_dir_prompt(args, console: Console) -> None:
    """Prompt for input directory selection."""
    if args.input_dir is None and not getattr(args, "visualize", False):
        card_dir = find_camera_card()

        choices = [
            questionary.Choice(
                title=f"Sort folder from PC ({TEST_DIR})",
                value=str(TEST_DIR),
            ),
            questionary.Choice(
                title=(
                    f"Memory card from camera ({card_dir})"
                    if card_dir
                    else "Memory card from camera"
                ),
                value=str(card_dir) if card_dir else None,
                disabled=None if card_dir else "no card mounted",
            ),
            questionary.Choice(
                title="Visualize an image with YOLO",
                value="visualize",
            ),
        ]

        result = _ask(questionary.select(
            "Select where to sort from (use arrow keys and Enter):",
            choices=choices,
            style=custom_style,
        ))

        if result is not None:
            if result == "visualize":
                args.visualize = True
                args.input_dir = None
            else:
                args.input_dir = result
                args.visualize = False
        else:
            console.print("No path selected. Exiting.")
            exit(1)


def interactive_model_prompt(args, console: Console) -> None:
    """Prompt for model selection."""
    if args.model is None:
        options = [
            {
                "filename": "yolo11x_custom.pt",
                "speed": "Slowest",
                "accuracy": "Highest",
                "use_case": "Trained on only relevant classes",
            },
            {
                "filename": "yolo11x.pt",
                "speed": "Slowest",
                "accuracy": "Highest",
                "use_case": "Best accuracy, slowest, big RAM",
            },
            {
                "filename": "yolo11s.pt",
                "speed": "Fast",
                "accuracy": "Good",
                "use_case": "Quick sorting",
            },
        ]

        # Table formatting
        col_model, col_speed, col_accuracy, col_use_case = 17, 10, 10, 30
        row_format = f"{{:<{col_model}}} | {{:<{col_speed}}} | {{:<{col_accuracy}}} | {{:<{col_use_case}}}"
        header = row_format.format("Model", "Speed", "Accuracy", "Use case")
        separator = " | ".join([
            "-" * col_model, "-" * col_speed, 
            "-" * col_accuracy, "-" * col_use_case
        ])

        choices = [
            questionary.Choice(title=header, value=None, disabled=" "),
            questionary.Choice(title=separator, value=None, disabled=" "),
        ]

        choices += [
            questionary.Choice(
                title="  " + row_format.format(
                    opt["filename"], opt["speed"], 
                    opt["accuracy"], opt["use_case"]
                ),
                value=opt["filename"],
            )
            for opt in options
        ]

        result = _ask(questionary.select(
            "Select a model (use arrow keys and Enter):",
            choices=choices,
            style=custom_style,
        ))

        if result is not None:
            args.model = result
        else:
            console.print("No model selected. Exiting.")
            exit(1)