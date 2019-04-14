# 
#  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
#  All rights reserved.
# 
#  This file is covered by the LICENSE.txt license file in the root directory.
# 

if ! type "pdftk" > /dev/null; then
    echo "Please install pdftk: sudo apt-get install pdftk"
    exit 1
fi

if ! type "dot" > /dev/null; then
    echo "Please install dot: sudo apt-get install graphviz"
    exit 1
fi

for f in *.dot; do
    dot -Tpdf $f | csplit --quiet --elide-empty-files --prefix=tmpx - "/%%EOF/+1" "{*}" && pdftk tmpx* cat output $f.pdf && rm -f tmpx*
done
