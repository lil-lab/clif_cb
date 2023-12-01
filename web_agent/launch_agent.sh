#!/usr/bin/env bash
cd ../cb_vin_feedback
exec python -m web_agent.agent_interface $1 $2 $3 $4 $5
