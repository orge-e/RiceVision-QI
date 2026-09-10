from ricevision_qi.app.core.config import load_profile_config, merge_config


def test_profile_config_loads_known_profile():
    config = load_profile_config("plastic_bag_rice")

    assert config["profile_name"] == "plastic_bag_rice"
    assert config["segmentation_backend"] == "opencv_watershed"
    assert config["segmentation_method"]


def test_missing_profile_falls_back_to_default():
    config = load_profile_config("does_not_exist")

    assert config["profile_name"] == "default"
    assert config["segmentation_backend"] == "opencv_watershed"


def test_merge_config_accepts_profile_name():
    config = merge_config({"profile": "debug_fast"})

    assert config["profile_name"] == "debug_fast"
    assert config["segmentation_backend"] in {
        "opencv_threshold",
        "opencv_watershed",
        "edge_enhanced_watershed",
    }
