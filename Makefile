# `make paper-example` builds a pinned Python 3.12 environment from
# requirements-paper.txt and runs the paper's Section 5 examples.
PAPER_VENV ?= .venv-paper

.PHONY: paper-example paper-tests check-oracles
$(PAPER_VENV)/.installed: requirements-paper.txt pyproject.toml
	uv venv --python 3.12 $(PAPER_VENV)
	uv pip install --python $(PAPER_VENV)/bin/python -r requirements-paper.txt
	uv pip install --python $(PAPER_VENV)/bin/python --no-deps -e .
	touch $@

paper-example: $(PAPER_VENV)/.installed
	$(PAPER_VENV)/bin/python examples/section5_example.py
	@echo
	$(PAPER_VENV)/bin/python examples/closed_form_vs_mc.py
	@echo
	$(PAPER_VENV)/bin/python examples/near_tie_example.py

paper-tests: $(PAPER_VENV)/.installed
	$(PAPER_VENV)/bin/python -m pytest -q

# Needs R with the `forecast` and `MCS` packages. Regenerates the R oracle
# fixtures and fails if they differ from the stored ones.
check-oracles:
	Rscript tests/_reference/gen_dm_oracle.R | cmp - tests/_reference/dm_test_oracle_values.py
	Rscript tests/_reference/gen_mcs_case.R 2>/dev/null | grep -v "Elapsed Time" | cmp - tests/_reference/mcs_oracle_r_output.txt
	@echo "R oracle fixtures reproduce."
