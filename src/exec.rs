use std::collections::VecDeque;
use std::fs;
use std::io::IsTerminal;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use crate::Options;
use crate::error::{KageError, Result};
use crate::graph::{BuildLog, EdgeId, Graph, Plan, StatCache};

#[derive(Debug, Clone)]
pub struct BuildResult {
    pub ran: usize,
    pub failed: usize,
    pub elapsed_ms: u128,
    pub nothing_to_do: bool,
}

struct Palette {
    on: bool,
}

impl Palette {
    fn new(force: Option<bool>) -> Self {
        let on = match force {
            Some(v) => v,
            None => std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal(),
        };
        Self { on }
    }

    fn wrap(&self, code: &str, s: &str) -> String {
        if self.on {
            format!("\x1b[{code}m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    fn dim(&self, s: &str) -> String {
        self.wrap("2", s)
    }
    fn bold(&self, s: &str) -> String {
        self.wrap("1", s)
    }
    fn red(&self, s: &str) -> String {
        self.wrap("31", s)
    }
    fn green(&self, s: &str) -> String {
        self.wrap("32", s)
    }
    fn cyan(&self, s: &str) -> String {
        self.wrap("36", s)
    }
    fn magenta(&self, s: &str) -> String {
        self.wrap("35", s)
    }
}

fn colorize_desc(pal: &Palette, instruction: &str, desc: &str) -> String {
    if instruction == "link" {
        pal.magenta(desc)
    } else {
        pal.cyan(desc)
    }
}

fn format_elapsed(ms: u128) -> String {
    if ms < 1000 {
        format!("{ms}ms")
    } else {
        format!("{:.3}s", ms as f64 / 1000.0)
    }
}

pub fn execute(
    graph: &Graph,
    plan: Plan,
    log: BuildLog,
    stat: StatCache,
    opts: &Options,
) -> Result<BuildResult> {
    let pal = Palette::new(opts.color);
    let start = Instant::now();
    let total = plan.dirty.len();

    if total == 0 {
        if !opts.quiet {
            eprintln!("{}", pal.dim("kage: nothing to do."));
        }
        return Ok(BuildResult {
            ran: 0,
            failed: 0,
            elapsed_ms: start.elapsed().as_millis(),
            nothing_to_do: true,
        });
    }

    if opts.explain && !opts.quiet {
        for &eid in &plan.dirty {
            if let Some(r) = &plan.reasons[eid] {
                eprintln!(
                    "{}",
                    pal.dim(&format!(
                        "kage: rebuild {}: {r}",
                        graph.edges[eid].description
                    ))
                );
            }
        }
    }

    let jobs = opts.jobs.max(1).min(total);
    let print = Mutex::new(());
    let counter = AtomicUsize::new(0);
    let failed_count = AtomicUsize::new(0);
    let keep_going = opts.keep_going;
    let dry_run = opts.dry_run;
    let verbose = opts.verbose;
    let quiet = opts.quiet;
    let width = total.to_string().len();

    let run_one = |eid: EdgeId,
                   log: &Mutex<BuildLog>,
                   stat: &Mutex<StatCache>|
     -> std::result::Result<(), String> {
        let edge = &graph.edges[eid];
        let idx = counter.fetch_add(1, Ordering::SeqCst) + 1;
        let label = if verbose {
            edge.command.clone()
        } else {
            edge.description.clone()
        };
        if !quiet {
            let _g = print.lock().unwrap();
            let prefix = pal.dim(&format!("[{idx:>width$}/{total}]"));
            eprintln!(
                "{} {}",
                prefix,
                colorize_desc(&pal, &edge.instruction, &label)
            );
        }
        if dry_run {
            return Ok(());
        }

        for &o in &edge.outs {
            let full = stat.lock().unwrap().full_path(&graph.nodes[o].path);
            if let Some(parent) = full.parent() {
                fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
            }
        }
        if let Some(df) = &edge.depfile {
            let full = stat.lock().unwrap().full_path(df);
            if let Some(parent) = full.parent() {
                fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
            }
        }

        let (code, output) = spawn_shell(&edge.command, &graph.workdir);
        if code != 0 {
            let mut msg = format!("FAILED: {}\n{}", edge.command, output);
            if !msg.ends_with('\n') {
                msg.push('\n');
            }
            return Err(msg);
        }

        {
            let mut log = log.lock().unwrap();
            for &o in &edge.outs {
                log.set(&graph.nodes[o].path, edge.command_hash);
            }
        }
        {
            let mut stat = stat.lock().unwrap();
            for &o in &edge.outs {
                stat.invalidate(&graph.nodes[o].path);
            }
            if let Some(df) = &edge.depfile {
                stat.invalidate(df);
            }
        }
        Ok(())
    };

    let log = Mutex::new(log);
    let stat = Mutex::new(stat);

    if jobs == 1 {
        let mut ready: VecDeque<EdgeId> = plan.ready.into();
        let mut wait = plan.wait;
        let dirty_set: Vec<bool> = {
            let mut d = vec![false; graph.edges.len()];
            for &e in &plan.dirty {
                d[e] = true;
            }
            d
        };
        while let Some(eid) = ready.pop_front() {
            match run_one(eid, &log, &stat) {
                Ok(()) => {
                    for &o in &graph.edges[eid].outs {
                        for &ce in &graph.nodes[o].consumers {
                            if dirty_set[ce] && wait[ce] > 0 {
                                wait[ce] -= 1;
                                if wait[ce] == 0 {
                                    ready.push_back(ce);
                                }
                            }
                        }
                    }
                }
                Err(msg) => {
                    failed_count.fetch_add(1, Ordering::SeqCst);
                    let _g = print.lock().unwrap();
                    eprint!("{}", pal.red(&msg));
                    if !keep_going {
                        break;
                    }
                }
            }
        }
    } else {
        run_parallel(
            graph,
            &plan,
            jobs,
            keep_going,
            &failed_count,
            &run_one,
            &log,
            &stat,
            &pal,
            &print,
        );
    }

    let failed = failed_count.load(Ordering::SeqCst);
    let ran = counter.load(Ordering::SeqCst);
    if !dry_run {
        let _ = log.lock().unwrap().save();
    }
    let elapsed_ms = start.elapsed().as_millis();
    if !quiet {
        if failed == 0 {
            eprintln!(
                "{}",
                pal.green(&format!(
                    "{} built {ran} {} in {}",
                    pal.bold("kage:"),
                    if ran == 1 { "target" } else { "targets" },
                    format_elapsed(elapsed_ms)
                ))
            );
        } else {
            eprintln!("{}", pal.red(&format!("kage: {failed} job(s) failed")));
        }
    }
    if failed > 0 {
        return Err(KageError::build(format!("{failed} job(s) failed")));
    }
    Ok(BuildResult {
        ran,
        failed,
        elapsed_ms,
        nothing_to_do: false,
    })
}

fn spawn_shell(command: &str, workdir: &Path) -> (i32, String) {
    #[cfg(windows)]
    let mut cmd = {
        let mut cmd = Command::new("cmd");
        cmd.arg("/C");
        cmd
    };

    #[cfg(not(windows))]
    let mut cmd = {
        let mut cmd = Command::new("/bin/sh");
        cmd.arg("-c");
        cmd
    };

    match cmd
        .arg(command)
        .current_dir(workdir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
    {
        Ok(o) => {
            let mut s = String::from_utf8_lossy(&o.stdout).into_owned();
            let err = String::from_utf8_lossy(&o.stderr);

            if !err.is_empty() {
                s.push_str(&err);
            }

            (o.status.code().unwrap_or(1), s)
        }

        Err(e) => (1, format!("failed to spawn shell: {e}\n")),
    }
}

fn run_parallel<F>(
    graph: &Graph,
    plan: &Plan,
    jobs: usize,
    keep_going: bool,
    failed_count: &AtomicUsize,
    run_one: &F,
    log: &Mutex<BuildLog>,
    stat: &Mutex<StatCache>,
    pal: &Palette,
    print: &Mutex<()>,
) where
    F: Fn(EdgeId, &Mutex<BuildLog>, &Mutex<StatCache>) -> std::result::Result<(), String> + Sync,
{
    let n_edges = graph.edges.len();
    let wait: Vec<AtomicUsize> = plan.wait.iter().map(|&w| AtomicUsize::new(w)).collect();
    let dirty_set: Vec<bool> = {
        let mut d = vec![false; n_edges];
        for &e in &plan.dirty {
            d[e] = true;
        }
        d
    };
    struct Shared {
        ready: Mutex<VecDeque<EdgeId>>,
        inflight: AtomicUsize,
        stop: AtomicBool,
        cv: Condvar,
    }
    let shared = Arc::new(Shared {
        ready: Mutex::new(plan.ready.iter().copied().collect()),
        inflight: AtomicUsize::new(0),
        stop: AtomicBool::new(false),
        cv: Condvar::new(),
    });

    std::thread::scope(|scope| {
        for _ in 0..jobs {
            let shared = Arc::clone(&shared);
            let wait = &wait;
            let dirty_set = &dirty_set;
            let graph = graph;
            scope.spawn(move || {
                loop {
                    let eid = {
                        let mut ready = shared.ready.lock().unwrap();
                        loop {
                            if let Some(e) = ready.pop_front() {
                                shared.inflight.fetch_add(1, Ordering::SeqCst);
                                break e;
                            }
                            if shared.inflight.load(Ordering::SeqCst) == 0
                                || (shared.stop.load(Ordering::SeqCst) && !keep_going)
                            {
                                return;
                            }
                            ready = shared.cv.wait(ready).unwrap();
                        }
                    };

                    let result = if shared.stop.load(Ordering::SeqCst) && !keep_going {
                        Ok(())
                    } else {
                        run_one(eid, log, stat)
                    };

                    match result {
                        Ok(()) => {
                            if !(shared.stop.load(Ordering::SeqCst) && !keep_going) {
                                for &o in &graph.edges[eid].outs {
                                    for &ce in &graph.nodes[o].consumers {
                                        if dirty_set[ce] {
                                            let prev = wait[ce].fetch_sub(1, Ordering::SeqCst);
                                            if prev == 1 {
                                                shared.ready.lock().unwrap().push_back(ce);
                                                shared.cv.notify_one();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Err(msg) => {
                            failed_count.fetch_add(1, Ordering::SeqCst);
                            {
                                let _g = print.lock().unwrap();
                                eprint!("{}", pal.red(&msg));
                            }
                            if !keep_going {
                                shared.stop.store(true, Ordering::SeqCst);
                            }
                        }
                    }
                    shared.inflight.fetch_sub(1, Ordering::SeqCst);
                    shared.cv.notify_all();
                }
            });
        }
    });
}

pub fn clean(graph: &Graph, opts: &Options) -> Result<BuildResult> {
    let pal = Palette::new(opts.color);
    let mut removed = 0usize;
    for edge in &graph.edges {
        for &o in &edge.outs {
            let p = if Path::new(&graph.nodes[o].path).is_absolute() {
                graph.nodes[o].path.clone()
            } else {
                graph
                    .workdir
                    .join(&graph.nodes[o].path)
                    .display()
                    .to_string()
            };
            if fs::remove_file(&p).is_ok() {
                removed += 1;
                if !opts.quiet {
                    eprintln!("{} {}", pal.dim("clean"), p);
                }
            }
        }
        if let Some(df) = &edge.depfile {
            let p = graph.workdir.join(df);
            let _ = fs::remove_file(p);
        }
    }
    let log = graph.workdir.join(".kage_log");
    let _ = fs::remove_file(&log);
    if !opts.quiet {
        eprintln!(
            "{}",
            pal.green(&format!("kage: removed {removed} output(s)"))
        );
    }
    Ok(BuildResult {
        ran: removed,
        failed: 0,
        elapsed_ms: 0,
        nothing_to_do: removed == 0,
    })
}

pub fn print_targets(graph: &Graph) {
    let mut outs: Vec<_> = graph.outputs().map(|s| s.to_string()).collect();
    outs.sort();
    outs.dedup();
    for o in outs {
        println!("{o}");
    }
}

pub fn print_compdb(graph: &Graph) {
    println!("[");
    let mut first = true;
    let dir = graph.workdir.display().to_string();
    for edge in &graph.edges {
        if edge.instruction == "link" {
            continue;
        }
        let file = edge
            .ins
            .first()
            .map(|&i| graph.nodes[i].path.clone())
            .unwrap_or_default();
        let output = edge
            .outs
            .first()
            .map(|&i| graph.nodes[i].path.clone())
            .unwrap_or_default();
        if !first {
            println!(",");
        }
        first = false;
        let cmd = json_escape(&edge.command);
        let file = json_escape(&file);
        let output = json_escape(&output);
        let dir = json_escape(&dir);
        print!(
            "  {{\"directory\": \"{dir}\", \"command\": \"{cmd}\", \"file\": \"{file}\", \"output\": \"{output}\"}}"
        );
    }
    println!("\n]");
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}
