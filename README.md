# computer-vision-cmake

This project aims to compile Kenneth Dawson-Howe's OpenCV Example code on non-Windows devices using CMake

## Build Dependencies

To build this project, you need:
- [CMake](https://cmake.org/download/)
- C++ build system (I'm using [Ninja](https://ninja-build.org/))
- C++ Compiler (I'm using [clang](https://clang.llvm.org/))
- [OpenCV](https://opencv.org/get-started/)
- a copy of the *Media/* folder from the [original source code](https://publications.scss.tcd.ie/book-supplements/A-Practical-Introduction-to-Computer-Vision-with-OpenCV/Code/). 

I've decided not to include the *Media/* folder directly because a lot of the video files are pretty large and not really suited for storing in a repository.

## Common Issues

- For some reason, some of the images in the *Media/* folder have the extension .JPG instead of .jpg. This messes up the program, so just rename all the file extensions with capital letters.

## Contributing

If you have any issues and want to submit a PR, just let me know and I'll review it (you can email me at mcgoffs@tcd.ie if you don't have my contact).