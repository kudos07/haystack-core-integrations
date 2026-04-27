# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from .document_store import VespaDocumentStore
from .filters import _normalize_filters

__all__ = ["VespaDocumentStore", "_normalize_filters"]
