#!/bin/bash
# Helper script to mount G: drive in WSL2
# Run this once before batch generation

echo "Mounting G: drive to /mnt/g..."
echo "You may be prompted for your sudo password."
echo ""

# Create mount point
sudo mkdir -p /mnt/g

# Mount the drive
sudo mount -t drvfs G: /mnt/g

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ G: drive mounted successfully!"
    echo ""
    echo "Testing access to dataset..."
    if [ -d "/mnt/g/My Drive/Archive - PhD/sa_video_story_engine" ]; then
        echo "✓ Dataset path accessible!"
        echo ""
        ls -lh "/mnt/g/My Drive/Archive - PhD/sa_video_story_engine" | head -15
    else
        echo "✗ Dataset path not found"
        echo "Please verify the path exists on G:"
    fi
else
    echo "✗ Failed to mount G: drive"
    echo "Please check if:"
    echo "  1. G: drive exists on Windows"
    echo "  2. You have proper permissions"
fi
