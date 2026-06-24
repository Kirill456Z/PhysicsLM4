#!/bin/bash

# Default image name (base name without tag)
DEFAULT_IMAGE="registry.rcp.epfl.ch/mlo-zemlians/physics4lm"

# Parse arguments
DELETE_REMOTE=false
IMAGE_INPUT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--remote)
            DELETE_REMOTE=true
            shift
            ;;
        *)
            IMAGE_INPUT="$1"
            shift
            ;;
    esac
done

# Use default if no image provided
if [ -z "$IMAGE_INPUT" ]; then
    IMAGE_INPUT="$DEFAULT_IMAGE"
fi

# Strip tag if provided to get base image name
BASE_IMAGE=$(echo "$IMAGE_INPUT" | sed 's/:.*$//')

echo "Searching for all tags of: $BASE_IMAGE"
if [ "$DELETE_REMOTE" = true ]; then
    echo "(Will also delete from remote registry)"
fi
echo ""

# Get all tags for this image, sorted by creation date
# Format: REPOSITORY:TAG
TAGS=$(docker images --format "{{.Repository}}:{{.Tag}}" --filter "reference=${BASE_IMAGE}" | grep -v "<none>")

if [ -z "$TAGS" ]; then
    echo "No images found for $BASE_IMAGE"
    exit 1
fi

echo "Found images:"
echo "$TAGS"
echo ""

# Get the most recently created image (first in docker images output, which is sorted by creation time)
LATEST_TAG=$(docker images --format "{{.Repository}}:{{.Tag}}" --filter "reference=${BASE_IMAGE}" | grep -v "<none>" | head -n 1)

if [ -z "$LATEST_TAG" ]; then
    echo "Could not determine latest tag"
    exit 1
fi

# Count total images
TOTAL_COUNT=$(echo "$TAGS" | wc -l | tr -d ' ')
OTHER_COUNT=$((TOTAL_COUNT - 1))

echo "Latest tag (most recently created): $LATEST_TAG"
echo ""

if [ "$OTHER_COUNT" -eq 0 ]; then
    echo "No other tags to delete. Only one version exists."
    exit 0
fi

echo "This will DELETE $OTHER_COUNT other tag(s), keeping only: $LATEST_TAG"
echo ""
echo "Tags to be deleted:"
echo "$TAGS" | grep -v "^${LATEST_TAG}$"
echo ""

# Prompt for confirmation
read -p "Are you sure you want to delete all tags except $LATEST_TAG? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Deleting old tags locally..."
    while IFS= read -r tag; do
        if [ "$tag" != "$LATEST_TAG" ]; then
            echo "Deleting $tag..."
            docker rmi "$tag"
        fi
    done <<< "$TAGS"
    echo ""
    echo "Local cleanup complete! Kept: $LATEST_TAG"
    
    # Delete from remote registry if requested
    if [ "$DELETE_REMOTE" = true ]; then
        echo ""
        echo "Deleting from remote registry..."
        
        # Extract registry URL and repository name
        REGISTRY=$(echo "$BASE_IMAGE" | cut -d'/' -f1)
        REPO_PATH=$(echo "$BASE_IMAGE" | cut -d'/' -f2-)
        
        # Get the tag part of LATEST_TAG
        LATEST_TAG_ONLY=$(echo "$LATEST_TAG" | sed 's/.*://')
        
        while IFS= read -r tag; do
            if [ "$tag" != "$LATEST_TAG" ]; then
                TAG_ONLY=$(echo "$tag" | sed 's/.*://')
                echo "Fetching digest for $TAG_ONLY..."
                
                # Get the manifest digest
                DIGEST=$(docker manifest inspect --verbose "$tag" 2>/dev/null | grep -m1 '"digest":' | cut -d'"' -f4)
                
                if [ -z "$DIGEST" ]; then
                    echo "Warning: Could not get digest for $tag, trying alternative method..."
                    # Alternative: use skopeo if available
                    if command -v skopeo &> /dev/null; then
                        DIGEST=$(skopeo inspect --raw docker://$tag | sha256sum | awk '{print "sha256:" $1}')
                    else
                        echo "Error: Cannot get digest for $tag. Install 'skopeo' or ensure image is pulled."
                        continue
                    fi
                fi
                
                echo "Deleting $tag (digest: $DIGEST) from remote registry..."
                
                # Use regctl if available (recommended)
                if command -v regctl &> /dev/null; then
                    regctl tag delete "$tag"
                # Use skopeo if available
                elif command -v skopeo &> /dev/null; then
                    skopeo delete docker://$tag
                else
                    echo "Warning: Neither 'regctl' nor 'skopeo' found."
                    echo "To delete from remote registry, install one of:"
                    echo "  - regctl: https://github.com/regclient/regclient"
                    echo "  - skopeo: https://github.com/containers/skopeo"
                    echo ""
                    echo "Or use Docker Registry API manually:"
                    echo "  curl -X DELETE https://$REGISTRY/v2/$REPO_PATH/manifests/$DIGEST"
                    break
                fi
            fi
        done <<< "$TAGS"
        
        echo ""
        echo "Remote cleanup complete!"
    fi
else
    echo "Deletion cancelled."
    exit 0
fi
