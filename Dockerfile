# コメントは#で始める

# 元にするdockerイメージをFROMコマンドでimportする
# 元にするdockerイメージはあらかじめDLしておく必要あり
# 元にするdockerイメージは基本、この下のやつにしとけば大丈夫
# 脳死でこれを使用
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# time zoneの設定(いるかわからんけどとりあえず)
ENV TZ=Asia/Tokyo
# pyenvのパスを通す(pyenv以外を使うのであればなくてよし)
ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"

# おまじない。この四行は必須と思ってok
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install tzdata

# ここからが本番
# 必要なものをapt-get install を使っていれていく
# RUNコマンドを通常のコマンドの前に入れれば、そのコマンドを使うことができる
# sudoはなくて大丈夫、僕は管理者らしい
# RUNコマンド内の改行は&& \ 
RUN apt-get -y install git make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev && \
    apt-get -y install wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python python3-pip

# workdirectryの設定
# ここは変えなくてよし
WORKDIR /workspace

# pytorchを入れる
#RUN pip3 install torch torchvision torchaudio torchtext --extra-index-url https://download.pytorch.org/whl/cu113 && \
# 使いたいパッケージを入れる
RUN pip3 install torchtext==0.14.1 && \
    pip3 install pytorch_lightning==1.7.7 && \
    pip3 install transformers==4.31.0 && \
    pip3 install pandas && \
    pip3 install undecorated && \
    pip3 install scikit-learn && \
    pip3 install scipy && \
    pip3 install evaluate && \
    pip3 install accelerate && \
    pip3 install nltk && \
    pip3 install whoosh  && \
    pip3 install ujson && \
    pip3 install numpy && \
    pip3 install matplotlib && \
    pip3 install flask && \
    pip3 install flask-socketio && \
    pip3 install datasketch && \
    pip3 install langdetect

# やっぱいらなかったものを消して軽くする
RUN apt-get autoremove -y

#COPY startup.sh /workspace
#RUN chmod 744 /startup.sh
#CMD ["/workspace/startup.sh"]