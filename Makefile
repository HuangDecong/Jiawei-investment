# 发布闸门：任一失败即阻断
#   make gate
.PHONY: gate test caliber audit deploy
gate: test caliber deploy

test:
	python -m pytest tests -q

caliber:
	python tools/check_caliber.py --all

deploy:
	python tools/deploy_check.py

audit:
	python tools/audit_all.py

vet:
	python -m pytest tests -q && python tools/audit_all.py && python tools/check_caliber.py
