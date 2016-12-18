import os
import yaml

profile_file = os.environ.get('PROFILE', 'default_profile.yaml')

with open(profile_file) as f:
    config = yaml.load(f)
