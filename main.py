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

# Import wasmtime trap to catch it specifically if needed
try:
    import wasmtime

    WasmTrap = wasmtime.Trap
except ImportError:
    WasmTrap = Exception  # Fallback if wasmtime isn't directly available

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ZarrReproducer")
# ---------------------

# --- Configuration ---
BASE_OUTPUT_DIR = pathlib.Path("./zarr_reproducer_output")
SMALL_SHAPE = (64, 1024, 1024)  # Small shape for ITKWASM test
LARGE_SHAPE = (1024, 2048, 2048)  # Large shape for TensorStore test
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


def test_itkwasm_gaussian_failure(
    input_array: np.ndarray, output_dir_base: pathlib.Path
):
    """
    Demonstrates the failure with Methods.ITKWASM_GAUSSIAN during multiscale generation.
    This failure occurs even with small data sizes due to internal WASM validation.
    """
    test_name = "itkwasm_gaussian_multiscale_failure"
    test_output_dir = output_dir_base / test_name
    scan_id = str(uuid.uuid4())[:8]  # Not strictly needed but keeps consistency

    logger.info(f"\n--- Running Test Case: {test_name} ---")
    logger.info(f"  Input Shape: {input_array.shape}")
    logger.info(f"  Multiscale Method: {Methods.ITKWASM_GAUSSIAN.name}")

    # Clean up previous run if exists
    if test_output_dir.exists():
        logger.warning(f"Removing existing directory: {test_output_dir}")
        shutil.rmtree(test_output_dir)
    test_output_dir.mkdir(parents=True, exist_ok=True)  # Create dir even if test fails

    start_time_total = time.time()
    try:
        # 1. Create NGFF Image
        logger.info("  Creating NGFF image object...")
        image = to_ngff_image(
            input_array,
            dims=("z", "y", "x"),
            scale={"z": PIXEL_SIZE, "y": PIXEL_SIZE, "x": PIXEL_SIZE},
            axes_units={"z": "micrometer", "y": "micrometer", "x": "micrometer"},
        )

        # 2. Create Multiscales (This is expected to fail)
        logger.info("  Creating multiscales (expecting failure)...")
        start_time_multiscale = time.time()
        # Use explicit chunks that previously triggered the issue, or None for default
        chunks_for_test = None
        multiscales = to_multiscales(
            image, method=Methods.ITKWASM_GAUSSIAN, chunks=chunks_for_test, cache=False
        )
        end_time_multiscale = time.time()
        logger.info(
            f"  Multiscales created in {end_time_multiscale - start_time_multiscale:.2f} seconds (UNEXPECTED SUCCESS)."
        )
        logger.warning(
            f"  UNEXPECTED SUCCESS: Test case '{test_name}' completed without the expected error."
        )
        return False  # Indicate unexpected success

    except WasmTrap as e:
        end_time_total = time.time()
        logger.info(
            f"  EXPECTED FAILURE: Test case '{test_name}' failed as expected after {end_time_total - start_time_total:.2f} seconds."
        )
        logger.info(f"  Caught expected WasmTrap: {e}")
        logger.exception(f"  Error details: {e}")
        return True  # Indicate expected failure occurred
    except Exception as e:
        end_time_total = time.time()
        logger.error(
            f"  UNEXPECTED FAILURE: Test case '{test_name}' failed after {end_time_total - start_time_total:.2f} seconds with an unexpected error."
        )
        logger.exception(f"  Error details: {e}")
        return False  # Indicate unexpected failure type


def test_tensorstore_already_exists_failure(
    input_array: np.ndarray, output_dir_base: pathlib.Path
):
    """
    Demonstrates the ALREADY_EXISTS failure with use_tensorstore=True during Zarr writing.
    This failure occurs with large data sizes that trigger regional writing.
    """
    test_name = "tensorstore_already_exists_write_failure"
    test_output_dir = output_dir_base / test_name
    scan_id = str(uuid.uuid4())[:8]
    zarr_path = test_output_dir / f"{scan_id}.zarr"

    logger.info(f"\n--- Running Test Case: {test_name} ---")
    logger.info(f"  Input Shape: {input_array.shape}")
    logger.info(
        f"  Multiscale Method: {Methods.DASK_IMAGE_GAUSSIAN.name} (using a working method)"
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
        # 1. Create NGFF Image
        logger.info("  Creating NGFF image object...")
        image = to_ngff_image(
            input_array,
            dims=("z", "y", "x"),
            scale={"z": PIXEL_SIZE, "y": PIXEL_SIZE, "x": PIXEL_SIZE},
            axes_units={"z": "micrometer", "y": "micrometer", "x": "micrometer"},
        )

        # 2. Create Multiscales (Using a method known to work)
        logger.info("  Creating multiscales...")
        start_time_multiscale = time.time()
        multiscales = to_multiscales(
            image,
            method=Methods.DASK_IMAGE_GAUSSIAN,  # Use a reliable method
            chunks=None,  # Default chunking
            cache=False,
        )
        end_time_multiscale = time.time()
        logger.info(
            f"  Multiscales created in {end_time_multiscale - start_time_multiscale:.2f} seconds."
        )

        # 3. Write Zarr (This is expected to fail)
        logger.info("  Writing NGFF Zarr with TensorStore (expecting failure)...")
        start_time_write = time.time()
        to_ngff_zarr(
            store=zarr_path,
            multiscales=multiscales,
            use_tensorstore=True,  # Enable TensorStore
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
        if "ALREADY_EXISTS" in str(e):
            logger.info(
                f"  EXPECTED FAILURE: Test case '{test_name}' failed as expected after {end_time_total - start_time_total:.2f} seconds."
            )
            logger.info(f"  Caught expected ValueError (ALREADY_EXISTS): {e}")
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

    # Test 1: ITKWASM_GAUSSIAN failure
    small_data = create_input_data(SMALL_SHAPE, INPUT_DTYPE, description="small")
    results["itkwasm_gaussian"] = test_itkwasm_gaussian_failure(
        small_data, BASE_OUTPUT_DIR
    )
    del small_data  # Free memory if needed

    # Test 2: TensorStore ALREADY_EXISTS failure
    large_data = create_input_data(LARGE_SHAPE, INPUT_DTYPE, description="large")
    results["tensorstore_already_exists"] = test_tensorstore_already_exists_failure(
        large_data, BASE_OUTPUT_DIR
    )
    del large_data  # Free memory

    logger.info("\n--- Reproducer Summary ---")
    logger.info(
        f"ITKWASM Gaussian Failure Test: {'EXPECTED FAILURE OCCURRED' if results.get('itkwasm_gaussian') else 'FAILED TO REPRODUCE / UNEXPECTED ERROR'}"
    )
    logger.info(
        f"TensorStore ALREADY_EXISTS Test: {'EXPECTED FAILURE OCCURRED' if results.get('tensorstore_already_exists') else 'FAILED TO REPRODUCE / UNEXPECTED ERROR'}"
    )
    logger.info("-------------------------")


if __name__ == "__main__":
    main()
