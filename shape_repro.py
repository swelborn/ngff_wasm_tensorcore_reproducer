import logging
import pathlib
import shutil
import time
import uuid

import numpy as np
from ngff_zarr import (
    Methods,
    to_multiscales,
    to_ngff_image,
    to_ngff_zarr,
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ZarrReproducer")
# ---------------------

# --- Configuration ---
BASE_OUTPUT_DIR = pathlib.Path("./zarr_reproducer_output")
LARGE_SHAPE = (1100, 2560, 2560)  # Large shape for TensorStore test
INPUT_DTYPE = np.float32
PIXEL_SIZE = 1.3
# ---------------------


def create_input_data(shape, dtype, description=""):
    """Generates random input data."""
    logger.info(
        f"Creating random {description} data with shape={shape}, dtype={dtype}..."
    )
    start_time = time.time()
    data = np.random.rand(*shape).astype(dtype)
    end_time = time.time()
    size_gb = data.nbytes / (1024**3)
    logger.info(
        f"Input data created ({size_gb:.3f} GB) in {end_time - start_time:.2f} seconds."
    )
    return data


def test_tensorstore_chunk_shape_failure(
    input_array: np.ndarray, output_dir_base: pathlib.Path
):
    test_name = "tensorstore_chunk_shape_write_failure"
    test_output_dir = output_dir_base / test_name
    scan_id = str(uuid.uuid4())[:8]
    zarr_path = test_output_dir / f"{scan_id}.zarr"

    logger.info(f"\n--- Running Test Case: {test_name} ---")
    logger.info(f"  Input Shape: {input_array.shape}")
    logger.info(
        f"  Multiscale Method: {Methods.ITKWASM_GAUSSIAN.name} (using a working method)"
    )
    logger.info("  Use TensorStore: True")
    logger.info(f"  Output Path: {zarr_path}")

    # Clean up previous run if exists
    if test_output_dir.exists():
        logger.warning(f"Removing existing directory: {test_output_dir}")
        shutil.rmtree(test_output_dir)
    test_output_dir.mkdir(parents=True)

    start_time_total = time.time()
    try:
        logger.info("  Creating NGFF image object...")
        image = to_ngff_image(
            input_array,
            dims=("z", "y", "x"),
            scale={"z": PIXEL_SIZE, "y": PIXEL_SIZE, "x": PIXEL_SIZE},
            axes_units={"z": "micrometer", "y": "micrometer", "x": "micrometer"},
        )

        logger.info("  Creating multiscales...")
        start_time_multiscale = time.time()
        multiscales = to_multiscales(
            image,
            method=Methods.ITKWASM_GAUSSIAN,
            cache=False,
        )
        end_time_multiscale = time.time()
        logger.info(
            f"  Multiscales created in {end_time_multiscale - start_time_multiscale:.2f} seconds."
        )

        logger.info("  Writing NGFF Zarr with TensorStore (expecting failure)...")
        start_time_write = time.time()
        to_ngff_zarr(
            store=zarr_path,
            multiscales=multiscales,
            use_tensorstore=True,
            chunks_per_shard=2,
            version="0.5",
        )
        end_time_write = time.time()
        logger.info(
            f"  Zarr written in {end_time_write - start_time_write:.2f} seconds (UNEXPECTED SUCCESS)."
        )
        logger.warning(
            f"  UNEXPECTED SUCCESS: Test case '{test_name}' completed without the expected error."
        )
        return False  # Indicate unexpected success

    except ValueError as e:
        end_time_total = time.time()
        if "FAILED_PRECONDITION:" in str(e):
            logger.info(
                f"  EXPECTED FAILURE: Test case '{test_name}' failed as expected after {end_time_total - start_time_total:.2f} seconds."
            )
            logger.info(f"  Caught expected ValueError (FAILED_PRECONDITION): {e}")
            logger.exception(f"  Error details: {e}")
            return True  # Indicate expected failure occurred
        else:
            logger.error(
                f"  UNEXPECTED FAILURE: Test case '{test_name}' failed after {end_time_total - start_time_total:.2f} seconds with an unexpected ValueError."
            )
            logger.exception(f"  Error details: {e}")
            return False  # Indicate unexpected failure type
    except Exception as e:
        end_time_total = time.time()
        logger.error(
            f"  UNEXPECTED FAILURE: Test case '{test_name}' failed after {end_time_total - start_time_total:.2f} seconds with an unexpected error."
        )
        logger.exception(f"  Error details: {e}")
        return False  # Indicate unexpected failure type


def main():
    """Runs the specific reproducer test cases."""
    # Prepare output directory
    if BASE_OUTPUT_DIR.exists():
        logger.warning(f"Removing base output directory: {BASE_OUTPUT_DIR}")
        shutil.rmtree(BASE_OUTPUT_DIR)
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # Test 2: TensorStore ALREADY_EXISTS failure
    large_data = create_input_data(LARGE_SHAPE, INPUT_DTYPE, description="large")
    results["tensorstore_chunk_shape"] = test_tensorstore_chunk_shape_failure(
        large_data, BASE_OUTPUT_DIR
    )
    del large_data  # Free memory

    logger.info("\n--- Reproducer Summary ---")
    logger.info(
        f"TensorStore ALREADY_EXISTS Test: {'EXPECTED FAILURE OCCURRED' if results.get('tensorstore_chunk_shape') else 'FAILED TO REPRODUCE / UNEXPECTED ERROR'}"
    )
    logger.info("-------------------------")


if __name__ == "__main__":
    main()
