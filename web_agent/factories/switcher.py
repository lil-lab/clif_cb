import subprocess
import yaml

from dataclasses import dataclass

from webapp.agent_factory.abstract_factory import abstract_factory
from webapp.agent_factory import abstract_agent_client
from webapp.utils import args
from typing import Dict


@dataclass
class FactoryArgs:
    agents: Dict[str, Dict[str, str]]


with open('../cb_vin_feedback/web_agent/factories/switcher.yml') as infile:
    parsed_yml = yaml.load(infile)
ARGS = FactoryArgs(**parsed_yml)
AGENT_NAMES = sorted(ARGS.agents.keys())


class FollowerSwitchingFactory(abstract_factory.AbstractFactory):
    def __init__(self):
        super(FollowerSwitchingFactory, self).__init__()

    def get_new_leader(self, webapp_config: args.WebappArgs):
        pass

    def get_new_follower(
            self, webapp_config: args.WebappArgs,
            agent_idx: int) -> abstract_agent_client.AbstractAgentClient:
        agent_uuid = self.get_random_agent_uuid()

        agent_type = AGENT_NAMES[agent_idx]
        agent_uuid = '%s_%s' % (agent_uuid, agent_type)

        print('Launching a %s agent' % agent_type)

        assert ARGS.agents[agent_type]['sampling'] in {'True', 'False'}

        process = subprocess.Popen([
            'sh', '../cb_vin_feedback/web_agent/launch_agent.sh', agent_uuid,
            ARGS.agents[agent_type]['experiment_name'],
            ARGS.agents[agent_type]['model_save'],
            str(webapp_config.port), ARGS.agents[agent_type]['sampling']
        ])

        return abstract_agent_client.SeparateProcessAgentClient(
            agent_uuid, process)
