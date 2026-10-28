# Page sketch — a scan with no text layer

This is the page a PDF extractor will see. It is not a document the agent should search.

Page 2 is a photograph of a facility sheet. The words "The vault PIN is 4419." are visible to a person and are not encoded as text.

Today's ingest can only mark that page with "no text layer". The PIN must not be copied into the search corpus. The golden file is what ingest should emit after OCR.
