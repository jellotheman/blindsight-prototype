# BlindSight vocabulary

BlindSight helps blind and low-vision people understand what they looked at during a short,
user-triggered camera capture. It describes the captured view without claiming complete or current
awareness of the surrounding place.

Read `REFERENCE.local.md` as well if it is present in your checkout.

**Capture:** An eight-second, user-triggered recording of whatever the camera sees while the user
looks or points it around. It has no required turn, pace, direction, or coverage target. Avoid
*scan*.

**Captured view:** The visual evidence observed during one capture. It is not a complete model of a
room and is not guaranteed to remain current.

**Scene card:** The revisable structured understanding produced from a captured view: place type
when evident, a concise overview, observed objects and relationships, people, directly observed
visual character, and claim-specific uncertainties.

**Scene session:** One scene card, its revisions, and the follow-up conversation grounded in it. A
new capture starts a new scene session. "Done," another capture, or application shutdown ends it.

**Orientation:** The concise first description: place type when evident, occupancy, and the few
dominant objects and relationships that help the user understand the captured view.

**Visual impression:** Directly observed colours, lighting, and materials. Style or atmosphere is
interpretation and belongs only in a qualified answer on request.

**Details on demand:** Follow-up questions inside a scene session. Nothing is deliberately hidden,
but only a short orientation is spoken unasked.

**Uncertainty:** What the system does not know, attached to the specific claim it qualifies.
Abstaining beats inventing.

**Silence:** A deliberate interaction state. During processing, silence is followed by bounded
audible progress cues so it cannot be mistaken for a crash.

**Attention budget:** The user's scarce auditory and cognitive capacity. Default speech is terse;
detail is user-controlled.

**World state:** The system's compressed understanding of what the camera sees at one moment. It is
not a scene card and is never spoken. Avoid *embedding*, *features*, *latent*.

**Transition:** A change of place large enough to justify a fresh understanding of the surroundings —
crossing a threshold, leaving a building, a street opening onto a square. Movement inside one place
is not a transition. Avoid *scene change*, *keyframe*, *event*.

**Transition event:** The system's judgement that a transition occurred. It offers the user the
chance to ask; it never starts a capture and never produces a description. Captures stay
user-triggered.

**Proactive descriptions:** An optional, off-by-default setting under which the system watches for
transitions continuously. The name is the user-facing setting; what the setting actually delivers is
an offer, not a description.
