FROM dsapp/triage:master

RUN echo 'export PS1="\[$(tput setaf 4)$(tput bold)[\]\u@$(tput setaf 6)$(tput smul)dirtyduck$(tput rmul)$(tput setaf 4)$:\\w]#\[$(tput sgr0) \]"' > /home/triage/.bashrc

USER root

RUN pip install ipython jupyter

USER triage

ENTRYPOINT [ "bash" ]
