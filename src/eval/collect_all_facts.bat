@echo off
set folder=%1

pushd %folder%

type train.txt facts.txt valid.txt test.txt > all.txt

popd
