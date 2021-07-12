```sh
cd neural_renderer
( cd ./docker/ && docker build . -t neural_renderer )

docker run --runtime=nvidia -it -v ${PWD}:/host neural_renderer
```
