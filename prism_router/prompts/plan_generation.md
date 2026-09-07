
## Plan Generation
When the task is complex (multi-step, involves design decisions, or requires
exploration), prepend a brief execution plan before the rewritten prompt:

## Plan
1. [Step 1 description]
2. [Step 2 description]
...

## Task
[Rewritten prompt from above]

Rules for plans:
- Only generate plans for genuinely complex tasks (not greetings, simple Q&A)
- Keep plans to 3-7 steps max
- Each step should be actionable, not vague
- The plan is a guide for the LLM, not a strict contract
- For simple/mid tier models: Skip plan generation, keep it simple
