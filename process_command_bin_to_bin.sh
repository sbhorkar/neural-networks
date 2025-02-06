#!/bin/bash

# This script takes multiple .bin file names as input and executes a command for each file
# after appending '-doubles' to the file name.

# Check if at least one input was provided
if [ $# -eq 0 ]
then
    echo "Usage: $0 <filename1.bin> <filename2.bin> ..."
    exit 1
fi

# Loop through all arguments
for bin_file in "$@"
do
    # Check if the file has a .bin extension
    if [[ $bin_file != *.bin ]]
    then
        echo "Warning: The file $bin_file does not have a .bin extension and will be skipped."
        continue
    fi

    # Append '-doubles' before the .bin extension
    modified_file="${bin_file/.bin/-doubles.bin}"

    # Execute the command with the modified file name
    echo "Processing $bin_file to $modified_file..."
    java Int2DoubleBin $bin_file $modified_file
    # Example command, replace with your actual command
    echo "command $modified_file"
done
