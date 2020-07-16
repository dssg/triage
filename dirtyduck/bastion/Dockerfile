FROM dsapp/triage:development

RUN echo 'export PS1="\[$(tput setaf 4)$(tput bold)[\]\u@$(tput setaf 2)$(tput smul)dirtyduck$(tput rmul)$(tput setaf 4)$:\\w]#\[$(tput sgr0) \]"' > /home/triage/.bashrc

USER triage

ENTRYPOINT [ "bash" ]
