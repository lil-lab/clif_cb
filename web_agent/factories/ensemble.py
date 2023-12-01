import subprocess
import yaml

from dataclasses import dataclass

from webapp.agent_factory.abstract_factory import abstract_factory
from webapp.agent_factory import abstract_agent_client
from webapp.utils import args
from typing import List


class FollowerEnsembleFactory(abstract_factory.AbstractFactory):
    def __init__(self):
        super(FollowerEnsembleFactory, self).__init__()

    def get_new_leader(self, webapp_config: args.WebappArgs):
        pass

    def get_new_follower(
        self, webapp_config: args.WebappArgs
    ) -> abstract_agent_client.AbstractAgentClient:
        agent_uuid = self.get_random_agent_uuid()

        process = subprocess.Popen([
            'sh', '../cb_vin_feedback/web_agent/launch_ensemble_agent.sh',
            agent_uuid, '../cb_vin_feedback/web_agent/factories/ensemble.yml',
            str(webapp_config.port)
        ])

        return abstract_agent_client.SeparateProcessAgentClient(
            agent_uuid, process)
