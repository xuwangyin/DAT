#!/bin/bash
# Setup script to download and extract datasets and model checkpoints using rclone
# Requires: rclone configured with gdrive and dropbox remotes

set -e  # Exit on error

setup_openimage() {
    local dir="./data/OpenImage-O-large"

    echo "Creating OpenImage-O-large directory..."
    mkdir -p "$dir"

    if [ -d "$dir/fid_eval" ]; then
        echo "fid_eval already exists, skipping"
    else
        echo "Downloading fid_eval.tar (download-first mode)..."
        rclone copy gdrive:Projects/robust-ebms-2/data/OpenImage-O-large/fid_eval.tar "$dir/" -P
        echo "Extracting fid_eval.tar..."
        tar -xf "$dir/fid_eval.tar" -C "$dir" --no-same-owner
        rm -f "$dir/fid_eval.tar"
    fi

    if [ -d "$dir/train" ]; then
        echo "train already exists, skipping"
    else
        echo "Downloading train.tar (download-first mode)..."
        rclone copy gdrive:Projects/robust-ebms-2/data/OpenImage-O-large/train.tar "$dir/" -P
        echo "Extracting train.tar..."
        tar -xf "$dir/train.tar" -C "$dir" --no-same-owner
        rm -f "$dir/train.tar"
    fi

    if [ -d "$dir/val" ]; then
        echo "val already exists, skipping"
    else
        echo "Downloading val.tar (download-first mode)..."
        rclone copy gdrive:Projects/robust-ebms-2/data/OpenImage-O-large/val.tar "$dir/" -P
        echo "Extracting val.tar..."
        tar -xf "$dir/val.tar" -C "$dir" --no-same-owner
        rm -f "$dir/val.tar"
    fi

    echo "OpenImage-O-large setup complete!"
}

setup_imagenet() {
    local dir="./data/ImageNet"
    if [ -d "$dir" ]; then
        echo "ImageNet already exists, skipping"
        return
    fi

    mkdir -p "$dir/train" "$dir/val"
    echo "Downloading ImageNet train files (parallel)..."
    rclone copy gdrive:Projects/robust-ebms-2/data/ImageNet/train "$dir/train" --transfers=8 --checkers=16 -P
    echo "Downloading ImageNet val.tar..."
    rclone copy gdrive:Projects/robust-ebms-2/data/ImageNet/val.tar "$dir/"
    echo "Downloading ImageNet metadata files..."
    rclone copy gdrive:Projects/robust-ebms-2/data/ImageNet/ILSVRC2012_devkit_t12.tar.gz "$dir/"
    rclone copy gdrive:Projects/robust-ebms-2/data/ImageNet/imagenet-simple-labels.json "$dir/"
    rclone copy gdrive:Projects/robust-ebms-2/data/ImageNet/meta.bin "$dir/"

    # Extract val.tar first
    echo "Extracting val.tar..."
    cd "$dir" && tar -xf val.tar && rm val.tar
    cd - > /dev/null

    # Extract tar files in parallel
    echo "Extracting training tar files..."
    cd "$dir/train"
    find . -name "*.tar" -print0 | xargs -0 -n1 -P8 -I {} bash -c '
        tar=${1}
        class=$(basename "$tar" .tar)
        mkdir -p "$class"
        tar -xf "$tar" -C "$class" && rm "$tar"
    ' _ {}
    cd - > /dev/null

    echo "Extracting validation tar files..."
    cd "$dir/val"
    find . -name "*.tar" -print0 | xargs -0 -n1 -P8 -I {} bash -c '
        tar=${1}
        class=$(basename "$tar" .tar)
        mkdir -p "$class"
        tar -xf "$tar" -C "$class" && rm "$tar"
    ' _ {}
    cd - > /dev/null

    echo "ImageNet setup complete!"
}

setup_checkpoints_and_fid() {
    echo "Downloading FID data and model checkpoints from Dropbox..."
    echo "Copying everything from dropbox:robust-ebms-2/ to project directory..."

    rclone copy dropbox:robust-ebms-2/ ./ --transfers=4 -P

    echo "Checkpoints and FID data setup complete!"
}

# Main execution
main() {
    echo "=========================================="
    echo "DAT Data Setup Script (using rclone)"
    echo "=========================================="
    echo ""

    # Check if rclone is installed
    if ! command -v rclone &> /dev/null; then
        echo "Error: rclone is not installed or not in PATH"
        echo "Please install rclone first: https://rclone.org/install/"
        exit 1
    fi

    # Check if remotes are configured
    if ! rclone listremotes | grep -q "gdrive:"; then
        echo "Warning: gdrive remote not configured in rclone"
        echo "Please configure it with: rclone config"
    fi

    if ! rclone listremotes | grep -q "dropbox:"; then
        echo "Warning: dropbox remote not configured in rclone"
        echo "Please configure it with: rclone config"
    fi

    echo ""
    echo "Select what to setup:"
    echo "1) OpenImage-O-large"
    echo "2) ImageNet"
    echo "3) Model checkpoints and FID data"
    echo "4) All of the above"
    echo ""
    read -p "Enter your choice (1-4): " choice

    case $choice in
        1)
            setup_openimage
            ;;
        2)
            setup_imagenet
            ;;
        3)
            setup_checkpoints_and_fid
            ;;
        4)
            setup_openimage
            echo ""
            setup_imagenet
            echo ""
            setup_checkpoints_and_fid
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    echo ""
    echo "=========================================="
    echo "Setup completed successfully!"
    echo "=========================================="
}

# Run main function if script is executed (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main
fi
