FROM golang:1.14 as builder
WORKDIR /go/src
RUN mkdir ml-tutorial-go
WORKDIR /go/src/ml-tutorial-go
RUN go get k8s.io/klog
RUN go get github.com/randysimpson/go-matrix/matrix
ADD . .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-extldflags "-static"' -o main .
FROM scratch
COPY --from=builder /go/src/ml-tutorial-go/main /app/
WORKDIR /app
CMD ["./main"]