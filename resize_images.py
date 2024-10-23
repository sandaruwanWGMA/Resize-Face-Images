import SimpleITK as sitk
import numpy as np


def resize_image(input_file_path, output_file_path=None, desired_image_path=None):
    # Read the input image
    input_image = sitk.ReadImage(input_file_path)

    if desired_image_path is None:
        raise ValueError("A desired_image_path must be provided.")

    # Read the desired image to get its size
    desired_image = sitk.ReadImage(desired_image_path)
    desired_size = desired_image.GetSize()

    input_size = input_image.GetSize()

    # Calculate the difference in size
    delta_width = desired_size[0] - input_size[0]
    delta_height = desired_size[1] - input_size[1]

    # Initialize crop and pad sizes
    crop_left = crop_right = 0
    crop_top = crop_bottom = 0
    pad_left = pad_right = 0
    pad_top = pad_bottom = 0

    # Determine cropping or padding for width
    if delta_width < 0:
        # Need to crop width
        crop_amount = -delta_width
        crop_left = int(crop_amount / 2)
        crop_right = int(crop_amount - crop_left)
    elif delta_width > 0:
        # Need to pad width
        pad_amount = delta_width
        pad_left = int(pad_amount / 2)
        pad_right = int(pad_amount - pad_left)

    # Determine cropping or padding for height
    if delta_height < 0:
        # Need to crop height
        crop_amount = -delta_height
        crop_top = int(crop_amount / 2)
        crop_bottom = int(crop_amount - crop_top)
    elif delta_height > 0:
        # Need to pad height
        pad_amount = delta_height
        pad_top = int(pad_amount / 2)
        pad_bottom = int(pad_amount - pad_top)

    # Perform cropping if needed
    if any([crop_left, crop_right, crop_top, crop_bottom]):
        cropped_image = sitk.Crop(
            input_image,
            lowerBoundaryCropSize=(crop_left, crop_top),
            upperBoundaryCropSize=(crop_right, crop_bottom),
        )
    else:
        cropped_image = input_image

    # Perform padding if needed
    if any([pad_left, pad_right, pad_top, pad_bottom]):
        # Convert to NumPy array to get border colors
        input_array = sitk.GetArrayFromImage(cropped_image)
        num_components = cropped_image.GetNumberOfComponentsPerPixel()

        # Get border colors
        if num_components == 1:
            # Grayscale image
            left_color = np.median(input_array[:, 0])
            right_color = np.median(input_array[:, -1])
            top_color = np.median(input_array[0, :])
            bottom_color = np.median(input_array[-1, :])
            border_color = np.median([left_color, right_color, top_color, bottom_color])
            pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
            padded_array = np.pad(
                input_array, pad_width, mode="constant", constant_values=border_color
            )
            # Convert back to SimpleITK image
            padded_image = sitk.GetImageFromArray(padded_array.astype(np.uint8))
        else:
            # Color image
            left_color = np.median(input_array[:, 0, :], axis=0)
            right_color = np.median(input_array[:, -1, :], axis=0)
            top_color = np.median(input_array[0, :, :], axis=0)
            bottom_color = np.median(input_array[-1, :, :], axis=0)
            border_color = np.median(
                [left_color, right_color, top_color, bottom_color], axis=0
            )
            pad_width = (
                (pad_top, pad_bottom),
                (pad_left, pad_right),
                (0, 0),
            )  # Do not pad the channels
            padded_array = np.pad(
                input_array, pad_width, mode="constant", constant_values=0
            )
            # Set the border color for each channel
            for c in range(num_components):
                if pad_top > 0:
                    padded_array[
                        :pad_top, pad_left : -pad_right if pad_right > 0 else None, c
                    ] = border_color[c]
                if pad_bottom > 0:
                    padded_array[
                        -pad_bottom:,
                        pad_left : -pad_right if pad_right > 0 else None,
                        c,
                    ] = border_color[c]
                if pad_left > 0:
                    padded_array[:, :pad_left, c] = border_color[c]
                if pad_right > 0:
                    padded_array[:, -pad_right:, c] = border_color[c]

            padded_image = sitk.GetImageFromArray(
                padded_array.astype(np.uint8), isVector=True
            )
    else:
        padded_image = cropped_image

    if output_file_path:
        sitk.WriteImage(padded_image, output_file_path)

    return padded_image
