```sh
cd neural_renderer
( cd ./docker/ && docker build . -t neural_renderer )

docker run -p 8888:8888 --runtime=nvidia -it -v ${PWD}:/host neural_renderer

# In container
cd /host
pip install -r ./docker/requirements_extra.txt
pip install --editable .
jupyter lab --allow-root --ip=0.0.0.0
```
