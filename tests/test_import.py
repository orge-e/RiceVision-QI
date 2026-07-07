from pathlib import Path


def test_project_modules_are_importable():
    import run_app
    from ricevision_qi.app.main import main
    from ricevision_qi.app.ui.main_window import MainWindow

    assert callable(run_app.main)
    assert callable(main)
    assert MainWindow.WINDOW_TITLE == "RiceVision-QI: Rice Quality Inspection System"


def test_detection_type_options_are_defined():
    from ricevision_qi.app.ui.main_window import DETECTION_TYPES

    assert DETECTION_TYPES == [
        "Rice Defective Kernel Inspection",
        "Paddy Husked Rice Yield",
        "Head Rice Yield",
    ]


def test_image_io_accepts_required_formats_case_insensitively():
    from ricevision_qi.app.core.image_io import is_supported_image_path

    supported = ["sample.jpg", "sample.PNG", "sample.bmp", "sample.Tif", "sample.TIFF"]

    for filename in supported:
        assert is_supported_image_path(Path(filename))


def test_image_io_rejects_unsupported_or_empty_paths():
    from ricevision_qi.app.core.image_io import is_supported_image_path

    assert not is_supported_image_path(Path("sample.gif"))
    assert not is_supported_image_path(Path(""))
