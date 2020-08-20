# ml-tutorial-go
Machine Learning tutorial in golang

## Introduction
This is meant to be a step by step guide to using machine learning algorithms in go.  It is assumed that you have basic understanding of golang.  I'll also be using docker as it will make compiling the projects much easier.  In fact I'm not even going to install golang on my machine, but you must have docker installed.  The process involves building the container and then running it to see the output.  To find out more read [Running module code with Docker](https://github.com/randysimpson/ml-tutorial-go#Running-module-code-with-Docker)

## Modules

* [01 - Linear Regression with SGD](https://github.com/randysimpson/ml-tutorial-go/blob/master/01_linear_regression_sgd/README.md)

## Running module code with Docker
All you need to do is clone this repo with `git clone https://github.com/randysimpson/ml-tutorial-go.git`.

```sh
~$ git clone https://github.com/randysimpson/ml-tutorial-go.git
Cloning into 'ml-tutorial-go'...
remote: Enumerating objects: 19, done.
remote: Counting objects: 100% (19/19), done.
remote: Compressing objects: 100% (14/14), done.
remote: Total 19 (delta 5), reused 19 (delta 5), pack-reused 0
Unpacking objects: 100% (19/19), done.
~$
```

Change directory to the repo `cd ml-tutorial-go` and then to the module `cd 01_linear_regression_sgd`.

```sh
~$ cd ml-tutorial-go
~/ml-tutorial-go$ cd 01_linear_regression_sgd
~/ml-tutorial-go/01_linear_regression_sgd$ 
```

Then create the container using the command `docker build -t randysimpson/ml-tutorial-go:v1.0 .`
```sh
~/ml-tutorial-go/01_linear_regression_sgd$ docker build -t randysimpson/ml-tutorial-go:v1.0 .
Sending build context to Docker daemon  3.072kB
Step 1/11 : FROM golang:1.14 as builder
 ---> a794da9351a3
Step 2/11 : WORKDIR /go/src
 ---> Using cache
 ---> fd93583b8d12
Step 3/11 : RUN mkdir ml-tutorial-go
 ---> Running in 3044df97fcd9
Removing intermediate container 3044df97fcd9
 ---> 0123dca661de
Step 4/11 : WORKDIR /go/src/ml-tutorial-go
 ---> Running in 46ca2ee1b1ea
Removing intermediate container 46ca2ee1b1ea
 ---> 0a9adbb7a2ec
Step 5/11 : RUN go get k8s.io/klog
 ---> Running in 1769cf45fd27
Removing intermediate container 1769cf45fd27
 ---> f9521860349f
Step 6/11 : ADD . .
 ---> 9f5bf9b295d2
Step 7/11 : RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-extldflags "-static"' -o main .
 ---> Running in 5ed72452beac
Removing intermediate container 5ed72452beac
 ---> f6ed11fa74d5
Step 8/11 : FROM scratch
 --->
Step 9/11 : COPY --from=builder /go/src/ml-tutorial-go/main /app/
 ---> c8353c7440b4
Step 10/11 : WORKDIR /app
 ---> Running in 2d1b11bc8cba
Removing intermediate container 2d1b11bc8cba
 ---> 5f023bc141ab
Step 11/11 : CMD ["./main"]
 ---> Running in bca37badfbd2
Removing intermediate container bca37badfbd2
 ---> 38e7b69d03ad
Successfully built 38e7b69d03ad
Successfully tagged randysimpson/ml-tutorial-go:v1.0
```

To run the container execute `docker run randysimpson/ml-tutorial-go:v1.0`
```sh
~/ml-tutorial-go/01_linear_regression_sgd$ docker run randysimpson/ml-tutorial-go:v1.0
I0819 16:13:14.933154       1 main.go:8] Initializing ml tutorial application
...
```

# Licence

MIT License

Copyright (c) 2020 Randy Simpson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.