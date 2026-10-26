# Decisions Model examples

Logical flow:
	1. Extract structured submission facts from the incoming flyer or submission.
	2. Run SAM’s existing inexpensive matching process.
	3. Send each candidate market plus the submission facts to Decision-1.
	4. Receive discrete rule scores, eligibility decisions, and flags.
	5. Calculate SAM’s weighted rule-group and final scores in deterministic application code.
	6. Rank the candidates.
	7. Optionally send only the top candidates to a larger model for a broker-facing explanation.
