SHELL= /bin/bash

SUBDIRS = ER_graph

.PHONY: subdirs $(SUBDIRS) erg

subdirs: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

erg:
	cp $(CURDIR)/ER_graph/er $(CURDIR)/main
