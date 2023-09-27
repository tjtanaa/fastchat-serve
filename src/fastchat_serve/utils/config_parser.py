from typing import Union, List
import json
import argparse

class MultiModelWorkerArgParser():

    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, "rb") as f:
            self.config = json.load(f)


    def override(self, args: argparse.Namespace) -> argparse.Namespace:
        args.model_path = self.model_path
        args.model_names = self.model_names
        args.languages = self.languages
        args.conv_template = self.conv_template
        args.host = self.host
        args.port = self.port
        args.worker_address = self.worker_address
        args.controller_address = self.controller_address
        return args
    
    @property
    def model_path(self):
        return self.config["model_path"]
    
    @property
    def model_names(self):
        return [self.config["model_names"]]
    
    @property
    def languages(self):
        return self.config["languages"]
    
    @property
    def conv_template(self):
        return self.config["conv_template"]
    
    @property
    def host(self):
        return self.config["host"]
    
    @property
    def port(self):
        return self.config["port"]

    @property
    def controller_address(self):
        return self.config["controller_address"]
    
    @property
    def worker_address(self):
        return self.config["worker_address"]