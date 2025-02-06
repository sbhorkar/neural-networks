#!/bin/bash

# This script takes multiple .bmp file names as input and executes a command for each file
# after converting the file name to a .bin extension.

# Check if at least one input was provided
if [ $# -eq 0 ]
then
    echo "Usage: $0 <filename1.bmp> <filename2.bmp> ..."
    exit 1
fi

# Loop through all arguments
for bmp_file in "$@"
do
    # Check if the file has a .bmp extension
    if [[ $bmp_file != *.bmp ]]
    then
        echo "Warning: The file $bmp_file does not have a .bmp extension and will be skipped."
        continue
    fi

    # Replace .bmp with .bin
    bin_file="${bmp_file/.bmp/.bin}"

    # Execute the command with the .bin file
    echo "Processing $bmp_file to $bin_file..."
    # java MyTest $bmp_file $bin_file
    java BMP2OneByte $bmp_file $bin_file
    # Example command, replace with your actual command
    echo "command $bin_file"
done
