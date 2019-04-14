
This directory contains example programs for using the compiler to compile code to the PUMA architecture.

Set up environment to point to libpuma.so:

    export LD_LIBRARY_PATH=`pwd`/../src:$LD_LIBRARY_PATH

Compile the examples:

    make                    # Compile all examples
    make <test-name>.test   # Compile a specific example

Execute the exxamples to generate the PUMA assembly code:

    ./<test-name>.test      # Execute a specific example

Generate PDF illustrations from .dot files (used for debugging)

    ./generate-pdf.sh

View compiler report

    cat <test-name>-report.out

