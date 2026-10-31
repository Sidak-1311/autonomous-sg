<!-- agentroom:start -->
## Working in the shared room

You are one of several AI agents working on this project at the same time, in a
shared room: the single source of truth for goals, specs, tasks and decisions.
The room hooks keep you in sync. They give you the room's state when your session
starts and other agents' updates between your steps; the latest state is also in
`.agentroom/ROOM.md`. Don't poll `get_room_state` or `get_conflicts`:
the hooks already bring you what's new.

1. Build only from what the room says. If the hooks gave you no room state, call
   `join_room` with the code in `.agentroom/room.json`, then `get_room_state`.
2. Claim as you go. Before each piece of work, call `announce_work` with the
   task and only the files it will touch, as paths relative to the repo root, not
   the whole project; your edits claim files as you make them too. If another
   agent in the room has the same goal, first propose a split with
   `publish_context`, or take the part their plan leaves you. If the room says
   another agent recently changed one of your files, pull or merge it first.
   Another window of your own agent (same name in the room) is a separate
   agent: split work with it the same way, and pass your `session` id on every
   room call so the room can tell you apart.
   Hold a file others also need (a router, a registry, shared config) only while
   you change it: then release it with `update_work` `release`, even though
   your task goes on, and at once when the room says someone is waiting for it.
3. When an update touches your task (a file you need is held, an interface you
   use changed, work you were waiting on is done, someone is doing the same
   thing), adjust right away. When you change direction or finish something
   others build on (an API, a schema, a shared helper), say so with
   `publish_context` then, not only at the end.
4. Code reaches other laptops only through git; the room only says what changed.
   When you finish something others build on, and before you mark a task done,
   commit and push it (`git pull --rebase` first if the push is rejected); the
   hooks hold `done` until it's pushed. When the room says a teammate pushed
   something you need, `git pull` before you use it or edit their files. If this
   folder isn't in git or has no remote, ask the user before you change anything,
   and work locally only if they say so.
5. Never wait idle. Blocked on another agent, do work no one holds, or call
   `get_room_updates` with `wait: 50`: it returns as soon as another agent
   does something, and `[]` means nothing yet, so call it again. Don't poll with
   `sleep`. Before you finish, check it once more: if another agent is still
   working toward your goal, wait the same way for up to 5 minutes for work they
   hand you. Say with `publish_context` when you start something others may
   use (a dev server, a port), and don't stop one another agent started.
6. Before a significant change (new dependency, schema or API change, deploy,
   deleting files, changing architecture), call `check_alignment`. Reading
   files and running tests change nothing and need no check. If it reports a
   conflict, don't proceed: call `raise_conflict` or ask the human. Never
   silently override another agent's work or the spec. Once a conflict is
   settled, call `resolve_conflict` with a note saying how.
7. Call `update_work` when the status changes and when you finish; the hooks
   report the files you edit.
8. Ask the human only when a conflict can't be resolved from the room's
   goals and specs.
<!-- agentroom:end -->
