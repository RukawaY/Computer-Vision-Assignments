#!/bin/bash

# Base directory containing all person-specific folders
BASE_DIR="../datasets"

# Check if the base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR not found."
    exit 1
fi

echo "Pre-Processing dataset..."

# Loop through each person's directory (e.g., S010, S011)
for PERSON_DIR in "$BASE_DIR"/S*; do
    if [ -d "$PERSON_DIR" ]; then
        PERSON_ID=$(basename "$PERSON_DIR")
        echo "Processing person: $PERSON_ID"

        # Create a list of all jpg images for this person from all subfolders (001, 002, etc.)
        IMAGE_LIST=()
        while IFS= read -r -d $'\0' file; do
            IMAGE_LIST+=("$file")
        done < <(find "$PERSON_DIR" -mindepth 2 -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) -print0)
        # The -mindepth 2 ensures we only look in subdirectories like 001, 002, etc.
        # and not directly in S010, S011, etc.

        NUM_IMAGES=${#IMAGE_LIST[@]}
        echo "Found $NUM_IMAGES images for $PERSON_ID"

        if [ "$NUM_IMAGES" -lt 10 ]; then
            echo "Warning: Not enough images for $PERSON_ID. Found $NUM_IMAGES, but need 10. Skipping this person."
            continue
        fi

        # Randomly select 20 images
        # Use gshuf if available, otherwise a simple array shuffle for portability if gshuf is not present
        SELECTED_IMAGES=()
        if command -v gshuf >/dev/null 2>&1; then
            TEMP_SELECTED_IMAGES=()
            while IFS= read -r line; do
                # Ensure line is not empty if the command output might have trailing newlines
                if [ -n "$line" ]; then
                    TEMP_SELECTED_IMAGES+=("$line")
                fi
            done < <(printf "%s\\n" "${IMAGE_LIST[@]}" | gshuf -n 20)

            # Check if gshuf actually returned 10 images
            if [ "${#TEMP_SELECTED_IMAGES[@]}" -eq 20 ]; then
                 SELECTED_IMAGES=("${TEMP_SELECTED_IMAGES[@]}")
            else
                echo "Warning: gshuf did not return 20 images for $PERSON_ID (got ${#TEMP_SELECTED_IMAGES[@]}). Attempting fallback."
                # Fallback if gshuf fails or returns less than 5 for some reason
                # This simple fallback might not be perfectly random for all cases but is a basic alternative.
                # For a truly robust random selection without gshuf, a more complex script logic would be needed.
                # Here we just take the first 5 after a sort -R (if sort supports it) or just first 5.
                TEMP_SELECTED_IMAGES=() # Reset for fallback
                while IFS= read -r line; do
                     if [ -n "$line" ]; then
                        TEMP_SELECTED_IMAGES+=("$line")
                    fi
                done < <(printf "%s\\n" "${IMAGE_LIST[@]}" | sort -R | head -n 5) # GNU sort

                if [ "${#TEMP_SELECTED_IMAGES[@]}" -eq 20 ]; then
                    SELECTED_IMAGES=("${TEMP_SELECTED_IMAGES[@]}")
                else
                    echo "Error: Fallback image selection also failed for $PERSON_ID (got ${#TEMP_SELECTED_IMAGES[@]}). Skipping."
                    continue
                fi
            fi
        else # Fallback if gshuf is not available
            echo "gshuf command not found, using alternative (less random) selection method."
            # Simple selection (first 5 after sorting - might not be very random)
            # For a better random selection without gshuf, one might implement Fisher-Yates shuffle in bash
            # or use other available tools like awk or python.
            # This is a placeholder for a more robust non-gshuf random selection.
            TEMP_SELECTED_IMAGES=()
            while IFS= read -r line; do
                if [ -n "$line" ]; then
                    TEMP_SELECTED_IMAGES+=("$line")
                fi
            done < <(printf "%s\\n" "${IMAGE_LIST[@]}" | head -n 10)

            if [ "${#TEMP_SELECTED_IMAGES[@]}" -lt 20 ] && [ "$NUM_IMAGES" -ge 20 ]; then # Ensure we get 20 if available
                 echo "Error: Alternative selection method failed to get 20 images for $PERSON_ID. Selected ${#TEMP_SELECTED_IMAGES[@]}. Skipping."
                 continue
            elif [ "${#TEMP_SELECTED_IMAGES[@]}" -lt 20 ]; then
                 echo "Warning: Not enough images selected by alternative method for $PERSON_ID. Found ${#TEMP_SELECTED_IMAGES[@]}, needed 20. Skipping."
                 continue
            fi
            SELECTED_IMAGES=("${TEMP_SELECTED_IMAGES[@]}")
        fi

        if [ "${#SELECTED_IMAGES[@]}" -ne 20 ]; then
            echo "Error: Could not select 20 images for $PERSON_ID. Selected ${#SELECTED_IMAGES[@]}. Skipping."
            continue
        fi

        echo "Selected 20 images for $PERSON_ID. Copying and renaming..."

        # Copy and rename the selected images
        COUNT=1
        for IMG_PATH in "${SELECTED_IMAGES[@]}"; do
            # Create the target directory for the person if it doesn't exist
            # This script assumes the target directory is the person's directory itself.
            TARGET_PERSON_DIR="$BASE_DIR/$PERSON_ID" # e.g. ./datasets/S010
            mkdir -p "$TARGET_PERSON_DIR"

            # Format number with leading zeros, e.g., 001, 002
            TARGET_FILENAME=$(printf "%03d.jpg" "$COUNT")
            TARGET_PATH="$TARGET_PERSON_DIR/$TARGET_FILENAME"

            cp "$IMG_PATH" "$TARGET_PATH"
            echo "Copied $IMG_PATH to $TARGET_PATH"
            COUNT=$((COUNT + 1))
        done

        # delete all subdirectories (001, 002, etc.)
        find "$TARGET_PERSON_DIR" -mindepth 1 -maxdepth 1 -type d -name "00*" -exec rm -rf {} +
        
        echo "Finished processing $PERSON_ID."
        echo "---"
    fi
done

echo "Dataset preparation complete."
echo "The selected images are now in $BASE_DIR/SXXX/001.jpg, $BASE_DIR/SXXX/002.jpg, etc."


echo "Selecting 40 random directories..."

# Get all directories in BASE_DIR
ALL_DIRS=()
for dir in "$BASE_DIR"/S*; do
    if [ -d "$dir" ]; then
        ALL_DIRS+=("$dir")
    fi
done

# Randomly select 40 directories if we have more than 40
if [ "${#ALL_DIRS[@]}" -gt 40 ]; then
    # Shuffle the array
    SHUFFLED_DIRS=("${ALL_DIRS[@]}")
    # Basic Fisher-Yates shuffle implementation for bash
    n=${#SHUFFLED_DIRS[@]}
    for ((i=0; i<n; i++)); do
        j=$((RANDOM % n))
        temp="${SHUFFLED_DIRS[$i]}"
        SHUFFLED_DIRS[$i]="${SHUFFLED_DIRS[$j]}"
        SHUFFLED_DIRS[$j]="$temp"
    done

    # Keep only first 40 directories
    SELECTED_DIRS=("${SHUFFLED_DIRS[@]:0:40}")

    # Remove other directories (those in ALL_DIRS but not in SELECTED_DIRS)
    for dir_to_check in "${ALL_DIRS[@]}"; do
        is_selected=false
        for selected_dir_path in "${SELECTED_DIRS[@]}"; do
            if [[ "$dir_to_check" == "$selected_dir_path" ]]; then
                is_selected=true
                break
            fi
        done

        if ! $is_selected; then
            echo "Removing directory (not selected): $dir_to_check"
            rm -rf "$dir_to_check"
        fi
    done
else
    SELECTED_DIRS=("${ALL_DIRS[@]}")
    echo "Warning: Found ${#ALL_DIRS[@]} directories, which is not more than 40. Using all available ${#ALL_DIRS[@]} directories."
fi

# Rename directories to S001, S002, etc. using a two-pass method
echo "Starting directory renaming process..."

TEMP_SUFFIX="__RENAMETEMP_$(date +%s%N)" # Unique temporary suffix
SELECTED_DIRS_ORIGINAL_ORDER=("${SELECTED_DIRS[@]}") # Preserve order for final naming
TEMP_DIRS_CREATED=() # Stores the actual temporary paths created after Pass 1

echo "Pass 1: Renaming selected directories to temporary unique names."
for original_dir in "${SELECTED_DIRS_ORIGINAL_ORDER[@]}"; do
    if [ -n "$original_dir" ] && [ -d "$original_dir" ]; then
        base_original_name=$(basename "$original_dir")
        if [ -z "$base_original_name" ] || [ "$base_original_name" == "." ] || [ "$base_original_name" == ".." ]; then
            echo "Warning (Pass 1): Invalid basename '$base_original_name' for directory '$original_dir'. Skipping."
            TEMP_DIRS_CREATED+=("ERROR_INVALID_BASENAME_SKIP")
            continue
        fi
        temp_path_target="$BASE_DIR/${base_original_name}${TEMP_SUFFIX}"

        echo "Pass 1: mv "$original_dir" "$temp_path_target""
        if mv "$original_dir" "$temp_path_target"; then
            TEMP_DIRS_CREATED+=("$temp_path_target")
        else
            echo "Error (Pass 1): Failed to rename '$original_dir' to '$temp_path_target'."
            TEMP_DIRS_CREATED+=("ERROR_DURING_RENAME_SKIP")
        fi
    else
        echo "Warning (Pass 1): Selected directory '$original_dir' not found or invalid. Skipping."
        TEMP_DIRS_CREATED+=("ERROR_ORIGINAL_NOT_FOUND_SKIP")
    fi
done

echo "Pass 2: Renaming temporary names to final S001, S002, ... format."
FINAL_COUNT=1
SUCCESSFUL_RENAMES=0
for temp_source_path in "${TEMP_DIRS_CREATED[@]}"; do
    if [[ "$temp_source_path" == "ERROR_"* ]]; then
        echo "Skipping final rename for an item due to error in Pass 1 ('$temp_source_path')."
        # This item will not be renamed to SXXX format, so the final count might be less.
        # We do not increment FINAL_COUNT here, as this SXXX slot will effectively be skipped.
        continue
    fi

    FINAL_NAME=$(printf "S%03d" "$FINAL_COUNT")
    FINAL_PATH="$BASE_DIR/$FINAL_NAME"

    if [ -d "$temp_source_path" ]; then
        echo "Pass 2: mv "$temp_source_path" "$FINAL_PATH""
        if mv "$temp_source_path" "$FINAL_PATH"; then
            SUCCESSFUL_RENAMES=$((SUCCESSFUL_RENAMES + 1))
        else
            echo "Error (Pass 2): Failed to rename '$temp_source_path' to '$FINAL_PATH'."
            # Even if it fails, we'll try the next SXXX number for the next valid temp dir.
        fi
    else
        # This indicates the temp directory, which should have been valid, is missing.
        echo "Critical Error (Pass 2): Temp source '$temp_source_path' (expected to be valid) not found for renaming to '$FINAL_PATH'."
    fi
    FINAL_COUNT=$((FINAL_COUNT + 1)) # Increment for the next SXXX name slot
done

echo "Directory processing complete."
if [ "${#SELECTED_DIRS_ORIGINAL_ORDER[@]}" -gt 0 ]; then
    echo "Attempted to process ${#SELECTED_DIRS_ORIGINAL_ORDER[@]} selected directories."
    echo "Successfully renamed $SUCCESSFUL_RENAMES directories into the S001-S$(printf "%03d" $((FINAL_COUNT-1))) format."
else
    echo "No directories were selected for processing."
fi


# CMAKE relative
echo "Making project..."

rm -rf build
mkdir build
cd build
cmake cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ..
make -j

# run prepare_data
./prepare_data

cd ..

# post-processing: keep only 7 images for each person, 5 for training, 2 for testing

rm -rf ../train_dataset
rm -rf ../test_dataset
mkdir ../train_dataset
mkdir ../test_dataset

for PERSON_DIR in ../datasets/S*; do
    if [ -d "$PERSON_DIR" ]; then
        PERSON_ID=$(basename "$PERSON_DIR")
        echo "Processing person: $PERSON_ID"
        # find all jpg images in the person's directory
        IMAGES=()
        while IFS= read -r -d $'\0' file ; do
            if [[ "$file" =~ _cropped\.jpe?g$ ]]; then
                IMAGES+=("$file")
            fi
        done < <(find "$PERSON_DIR" -mindepth 1 -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) -print0)

        # keep only 7 images
        if [ "${#IMAGES[@]}" -gt 7 ]; then
            # select first 7 images
            SELECTED_IMAGES=("${IMAGES[@]:0:7}")
        else
            SELECTED_IMAGES=("${IMAGES[@]}")
        fi

        # rename the selected images to 001.jpg, 002.jpg, etc.
        COUNT=1
        for IMG_PATH in "${SELECTED_IMAGES[@]}"; do
            mv "$IMG_PATH" "$PERSON_DIR/00$COUNT.jpg"
            COUNT=$((COUNT + 1))
        done

        mkdir ../train_dataset/$PERSON_ID
        mkdir ../test_dataset/$PERSON_ID

        # move the first 5 images to ../train_dataset
        mv "$PERSON_DIR/001.jpg" "$PERSON_DIR/002.jpg" "$PERSON_DIR/003.jpg" "$PERSON_DIR/004.jpg" "$PERSON_DIR/005.jpg" ../train_dataset/$PERSON_ID
        # move the last 2 images to ../test_dataset and rename to 001.jpg and 002.jpg
        mv "$PERSON_DIR/006.jpg" ../test_dataset/$PERSON_ID/001.jpg
        mv "$PERSON_DIR/007.jpg" ../test_dataset/$PERSON_ID/002.jpg
        
        echo "Finished processing $PERSON_ID."
        echo "---"
    fi
done

rm -rf ../datasets

echo "Post-processing complete."

echo "Dataset preparation complete."