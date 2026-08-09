# Stage 4B.4 Deployed Runtime Parity

- parity: `True`
- source_delivery_mode: `IMAGE_COPY`
- api_container_id: `6f94375ec4cae853ec2a0e93ac82d2253aa8cb5e5bf5e2f508cc446f7303dbf0`
- api_image_id: `sha256:b118925ce303af29be3564511cd1def72d728bf1d3194bf8a8555358382a12e8`
- old_container_predated_fix: `True`

| module | host sha | deployed sha | match | loaded path |
| --- | --- | --- | --- | --- |
| research_route | d9076e6856b45f63b96d5d96ea35fe4d6a03c9dbc58b74e44538c1330a55a254 | d9076e6856b45f63b96d5d96ea35fe4d6a03c9dbc58b74e44538c1330a55a254 | True | /usr/local/lib/python3.12/site-packages/paper_research/api/routes/research.py |
| agent_runner | cf4e8ce2dbe40935eb7ee5234190bfa8cf9176f291b31da547a0d2868fb195a2 | cf4e8ce2dbe40935eb7ee5234190bfa8cf9176f291b31da547a0d2868fb195a2 | True | /usr/local/lib/python3.12/site-packages/paper_research/agents/research_agent/runner.py |
| decision_provider | ffbf6168acbec915b6c816c46b3470c95e7c876e7111bafc17032c3692cd4772 | ffbf6168acbec915b6c816c46b3470c95e7c876e7111bafc17032c3692cd4772 | True | /usr/local/lib/python3.12/site-packages/paper_research/agents/research_agent/decision_provider.py |
| agent_checkpoint | 260502869dd4e31784f004eb185709971e7ca35e4347e655c3c51ca01e19b56f | 260502869dd4e31784f004eb185709971e7ca35e4347e655c3c51ca01e19b56f | True | /usr/local/lib/python3.12/site-packages/paper_research/agents/research_agent/checkpoint.py |
