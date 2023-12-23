FROM rayproject/ray:2.9.0-py310

USER root

ARG tpu

RUN apt-get upgrade -y && apt-get update -y && \
  apt-get install -y --upgrade python3 python3-pip python-is-python3 \
  coreutils rsync openssh-client curl

RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

RUN pip install cryptography
RUN pip install memray py-spy
RUN if [ -z "$tpu" ] ; then pip install "jax[cpu]==0.4.12" ; else pip install "jax[tpu]==0.4.12" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html ; fi
RUN pip install google-api-python-client google-cloud-storage
RUN pip install dm-reverb[tensorflow]==0.13.0
RUN pip install jaxlib==0.4.12
RUN pip install ipython
RUN pip install dm-env==1.6
RUN pip install 'gym==0.25.0'
RUN pip install gym[atari]==0.25.0
RUN pip install gym[accept-rom-license]==0.25.0
RUN pip install chex==0.1.82
RUN pip install rlax==0.1.6
RUN pip install numpy==1.23.5
RUN pip install pydantic==1.10.9
RUN pip install flax==0.6.11
RUN pip install Pillow==10.1.0
RUN pip install ml-collections==0.1.1
RUN pip install ml-dtypes==0.2.0
RUN pip install tensorboardx==2.6.2.2
RUN pip install orbax-checkpoint==0.4.1
RUN pip install tensorflow-cpu==2.15.0