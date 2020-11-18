# This Dockerfile creates an environment suitable for downstream usage of AllenNLP
# with allennlp-models. It's built from official release images of AllenNLP with a wheel
# install of the corresponding version of allennlp-models, and is used to publish
# the official allennlp/models images.
#
# This is very similar to Dockerfile.release, except that allennlp-models isn't
# installed from PyPI, it's installed from an arbitrary wheel build.
# The reason for this difference is that this image is built during a release workflow
# on a GitHub Actions job, which is required to succeed *before* the PyPI release is uploaded.

ARG ALLENNLP_TAG

FROM allennlp/allennlp:${ALLENNLP_TAG}

# Install the wheel of allennlp-models.
COPY dist dist/
RUN pip install --no-cache-dir $(ls dist/*.whl) && rm -rf dist/
