#!/bin/bash
set -x
set -e

if [ ! -d "searcher" ]; then
    echo 'Please run this script in the root directory of the project'
    exit 1
fi

if [ ! -f .env ]; then
    echo "Please create a .env file by using `cp .env.example .env`"
    exit 1
fi

# Read the current version from the .env file
VERSION=$(grep VERSION .env | cut -d= -f2)

# Split the version number into its components
IFS='.' read -r -a VERSION_ARRAY <<< "$VERSION"

# Increase the last component by 1
((VERSION_ARRAY[${#VERSION_ARRAY[@]}-1]++))

# Join the components back into a version number
NEW_VERSION=$(printf "%s." "${VERSION_ARRAY[@]}" | sed 's/\.$//')

# Replace the old version with the new version in the .env file
sed -i "s/VERSION=$VERSION/VERSION=$NEW_VERSION/g" .env

echo "Version increased from $VERSION to $NEW_VERSION"