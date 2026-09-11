The feedback given by the rewriter_v2 judge model (qwen3:14b), including alternative careers.

These are json files per job per rewritten candidate CV. 
Each .json includes:

{{
  "candidate": "{submission.candidate}",
  "job_title": "{submission.job_title}",
  "recommended_rewrite": "<exactly one of {agent_names}>",
  "recommendation_reason": "<3-6 sentences comparing the rewrites against the job listing and \
justifying the pick, referencing the fact-check results where relevant>",
  "career_roadmap": [
    {{"step": "<short name for this step, e.g. 'Apply for the role with Rewrite B'>",
     "action": "<what the candidate should concretely do for this step>"}},
    {{"step": "<next step>", "action": "<what to do>"}}
  ],
  "alternative_careers": ["<a different role or career path, distinct from '{submission.job_title}'>", "<another one>"]
}}
