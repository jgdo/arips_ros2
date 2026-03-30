FROM ros:kilted

ARG USERNAME=rosuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Remove existing user/group with UID/GID 1000 if present
RUN if getent passwd 1000 > /dev/null 2>&1; then userdel -r $(getent passwd 1000 | cut -d: -f1); fi \
    && if getent group 1000 > /dev/null 2>&1; then groupdel $(getent group 1000 | cut -d: -f1); fi

# Create a non-root user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN sudo usermod -a -G dialout $USERNAME

# Install additional packages
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        git \
        bash-completion \
        python3-pip \
        pipx \
        python3-serial \
    && rm -rf /var/lib/apt/lists/*

ENV SETUP_BASH=/opt/ros/kilted/setup.bash
RUN echo "source $SETUP_BASH" >> /home/$USERNAME/.bashrc
RUN echo "export PATH=\$PATH:/home/$USERNAME/.local/bin" >> /home/$USERNAME/.bashrc
RUN echo "if [ -f /home/$USERNAME/ws/install/setup.bash ]; then source /home/$USERNAME/ws/install/setup.bash; fi" >> /home/$USERNAME/.bashrc

ENV PYTHONPYCACHEPREFIX=/home/$USERNAME/.pycache

USER $USERNAME
COPY --chown=$USERNAME:$USERNAME colcon_defaults.yaml /home/$USERNAME/ws/colcon_defaults.yaml
WORKDIR /home/$USERNAME/ws
