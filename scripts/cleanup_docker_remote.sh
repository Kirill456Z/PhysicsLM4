#!/bin/bash

# Script to delete old Docker image tags from a remote registry
# Uses Docker Registry HTTP API v2

# Default image name
DEFAULT_IMAGE="registry.rcp.epfl.ch/mlo-zemlians/physics4lm"

# Get image name from argument or use default
IMAGE_INPUT="${1:-$DEFAULT_IMAGE}"

# Strip tag if provided to get base image name
BASE_IMAGE=$(echo "$IMAGE_INPUT" | sed 's/:.*$//')

# Extract registry and repository
REGISTRY=$(echo "$BASE_IMAGE" | cut -d'/' -f1)
REPO_PATH=$(echo "$BASE_IMAGE" | cut -d'/' -f2-)

echo "Fetching tags from remote registry: $REGISTRY"
echo "Repository: $REPO_PATH"
echo ""

# Get authentication token
# First, try to get credentials from docker config
DOCKER_CONFIG="$HOME/.docker/config.json"
AUTH_HEADER=""

if [ -f "$DOCKER_CONFIG" ]; then
    # Try to extract auth token from docker config
    AUTH_BASE64=$(jq -r ".auths[\"$REGISTRY\"].auth // empty" "$DOCKER_CONFIG" 2>/dev/null)
    if [ -n "$AUTH_BASE64" ]; then
        AUTH_HEADER="Authorization: Basic $AUTH_BASE64"
    fi
fi

# Fetch all tags from registry
echo "Fetching tags..."
TAGS_JSON=$(curl -s -H "$AUTH_HEADER" "https://$REGISTRY/v2/$REPO_PATH/tags/list")

if [ -z "$TAGS_JSON" ] || echo "$TAGS_JSON" | grep -q "errors"; then
    echo "Error: Could not fetch tags from registry."
    echo "Response: $TAGS_JSON"
    echo ""
    echo "You may need to login first:"
    echo "  docker login $REGISTRY"
    exit 1
fi

# Extract tags array
REMOTE_TAGS=$(echo "$TAGS_JSON" | jq -r '.tags[]' 2>/dev/null | sort -V)

if [ -z "$REMOTE_TAGS" ]; then
    echo "No tags found in remote registry"
    exit 1
fi

echo "Remote tags found:"
echo "$REMOTE_TAGS"
echo ""

# Find the latest tag (highest version number)
LATEST_TAG=$(echo "$REMOTE_TAGS" | tail -n 1)

echo "Latest tag (by version sort): $LATEST_TAG"
echo ""

# Count tags
TOTAL_COUNT=$(echo "$REMOTE_TAGS" | wc -l | tr -d ' ')
OTHER_COUNT=$((TOTAL_COUNT - 1))

if [ "$OTHER_COUNT" -eq 0 ]; then
    echo "No other tags to delete. Only one version exists."
    exit 0
fi

echo "This will DELETE $OTHER_COUNT other tag(s) from the remote registry."
echo "Keeping only: $LATEST_TAG"
echo ""
echo "Tags to be deleted:"
echo "$REMOTE_TAGS" | grep -v "^${LATEST_TAG}$"
echo ""

# Prompt for confirmation
read -p "Are you sure you want to delete these tags from $REGISTRY? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deletion cancelled."
    exit 0
fi

echo "Deleting tags from remote registry..."
echo ""

while IFS= read -r tag; do
    if [ "$tag" != "$LATEST_TAG" ]; then
        echo "Processing $tag..."
        
        # Get the manifest digest (must use Accept header for digest)
        DIGEST=$(curl -s -I -H "$AUTH_HEADER" \
            -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
            "https://$REGISTRY/v2/$REPO_PATH/manifests/$tag" \
            | grep -i "docker-content-digest:" | awk '{print $2}' | tr -d '\r')
        
        if [ -z "$DIGEST" ]; then
            echo "Error: Could not get digest for $tag"
            continue
        fi
        
        echo "  Digest: $DIGEST"
        echo "  Deleting..."
        
        # Delete the manifest
        DELETE_RESPONSE=$(curl -s -w "\n%{http_code}" -X DELETE \
            -H "$AUTH_HEADER" \
            "https://$REGISTRY/v2/$REPO_PATH/manifests/$DIGEST")
        
        HTTP_CODE=$(echo "$DELETE_RESPONSE" | tail -n 1)
        
        if [ "$HTTP_CODE" = "202" ] || [ "$HTTP_CODE" = "200" ]; then
            echo "  ✓ Successfully deleted $tag"
        else
            echo "  ✗ Failed to delete $tag (HTTP $HTTP_CODE)"
            echo "  Response: $(echo "$DELETE_RESPONSE" | head -n -1)"
        fi
        echo ""
    fi
done <<< "$REMOTE_TAGS"

echo "Remote cleanup complete!"
echo ""
echo "Note: The registry may need to run garbage collection to free up storage."
echo "Contact your registry administrator if needed."
