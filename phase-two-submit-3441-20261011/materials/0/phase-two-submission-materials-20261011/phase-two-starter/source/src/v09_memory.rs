//! Bounded transaction-level memory system. Addresses and transfers are 64-byte
//! lines. The caller owns numerical values and local RF/SH writeback; this module
//! owns only timing tags, shared arbitration and memory energy traffic.
//! Cache miss merging is unconditional when cache is installed; multicast adds
//! shared-trunk return routing, never free endpoint delivery. NoC byte counters
//! count trunk and endpoint bytes separately (one pJ per byte on each link).
use serde::Serialize;
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
};

#[derive(Clone, Debug)]
pub struct MemoryConfig {
    pub sms: usize,
    pub hbm_channels: usize,
    pub hbm_queue_depth: usize,
    pub sm_noc_bytes_per_cycle: usize,
    pub noc_bytes_per_cycle: usize,
    pub cache_mib: usize,
    pub cache_mshrs: usize,
    pub cache_bytes_per_cycle: usize,
    pub cache_latency: u64,
    pub multicast: bool,
    /// All stages together, including merged destinations and return traffic.
    pub max_pending_per_sm: usize,
}
impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            sms: 4,
            hbm_channels: 4,
            hbm_queue_depth: 64,
            sm_noc_bytes_per_cycle: 64,
            noc_bytes_per_cycle: 256,
            cache_mib: 0,
            cache_mshrs: 32,
            cache_bytes_per_cycle: 64,
            cache_latency: 20,
            multicast: false,
            max_pending_per_sm: 32,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    pub sm: usize,
    pub group: u64,
    pub token: u64,
    pub address: u64,
    pub valid_bytes: u32,
    pub write: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Completion {
    pub request: Request,
    pub cycle: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EnergyKind {
    HbmRead,
    HbmWrite,
    HbmTurnaround,
    Noc,
    CacheRead,
    CacheFill,
    CacheInvalidate,
}
#[derive(Clone, Copy, Debug, Serialize)]
pub struct EnergyEvent {
    pub cycle: u64,
    pub kind: EnergyKind,
    pub bytes: u64,
    pub duration: u64,
    pub energy_pj: f64,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct Stats {
    pub submitted: u64,
    pub completed: u64,
    pub valid_bytes: u64,
    pub hbm_reads: u64,
    pub hbm_writes: u64,
    pub noc_bytes: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub merged_reads: u64,
    pub miss_merged: u64,
    pub multicast_destinations: u64,
    pub noc_global_bytes: u64,
    pub noc_endpoint_bytes: u64,
    pub direction_switches: u64,
    pub max_pending: usize,
    pub max_hbm_inflight: usize,
    pub max_cache_mshrs: usize,
    pub max_events: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Event {
    Lookup(u64),
    HbmReady(u64),
    HbmDone(u64),
    HbmStart(u64),
    Turnaround,
    Fill(u64),
    WriteLookup(u64),
    Arrive(u64),
    NocTick,
    Fanout(u64),
    CacheTick,
}
#[derive(Clone)]
struct Transfer {
    request: Request,
    destinations: Vec<Request>,
}
struct Channel {
    waiting: VecDeque<u64>,
    used: usize,
    free: u64,
    direction: Option<bool>,
}
#[derive(Clone, Copy)]
struct Packet {
    id: u64,
    sm: usize,
    remaining: usize,
    /// 0=unicast both links, 1=multicast trunk, 2=fanout endpoint.
    mode: u8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CacheEntry {
    tag: u64,
    stamp: u64,
    valid: bool,
}
#[derive(Clone, Copy)]
struct LineUse {
    reads: usize,
    writes: usize,
}
/// Memory is driven at event boundaries. `advance_to` returns all completions up
/// to a scheduler time; callers should consume energy after every advance to keep
/// their trace bounded. Outstanding engine state is bounded by sms*frontend slots.
pub struct Memory {
    config: MemoryConfig,
    now: u64,
    serial: u64,
    events: BinaryHeap<Reverse<(u64, u64, Event)>>,
    transfers: HashMap<u64, Transfer>,
    channels: Vec<Channel>,
    pending: Vec<usize>,
    pending_total: usize,
    line_use: HashMap<u64, LineUse>,
    merge: HashMap<u64, u64>,
    cache: Vec<[CacheEntry; 4]>,
    stamp: u64,
    miss_wait: VecDeque<u64>,
    mshrs: usize,
    cache_queue: VecDeque<(u64, usize, u8)>,
    cache_tick: Option<u64>,
    noc_queue: VecDeque<Packet>,
    noc_tick: Option<u64>,
    // Delivery IDs name individual destinations, including multicast fanout.
    deliveries: HashMap<u64, (Request, Option<u64>)>,
    fanouts: HashMap<u64, Vec<(Request, Option<u64>)>>,
    acknowledgements: HashMap<u64, (Request, Option<u64>)>,
    tickets: HashMap<u64, (usize, usize)>,
    tokens: HashSet<u64>,
    energy: Vec<EnergyEvent>,
    stats: Stats,
}
impl Memory {
    pub fn new(config: MemoryConfig) -> Result<Self, String> {
        if config.sms == 0
            || config.sms > 32
            || ![1, 2, 4, 8].contains(&config.hbm_channels)
            || config.hbm_queue_depth == 0
            || config.hbm_queue_depth > 512
            || config.sm_noc_bytes_per_cycle == 0
            || config.sm_noc_bytes_per_cycle > 512
            || config.noc_bytes_per_cycle == 0
            || config.noc_bytes_per_cycle > 4096
            || ![0, 1, 2, 4, 8, 16].contains(&config.cache_mib)
            || (config.cache_mib > 0 && config.cache_mshrs == 0)
            || config.cache_mshrs > 4096
            || (config.cache_mib > 0 && config.cache_bytes_per_cycle == 0)
            || config.cache_bytes_per_cycle > 4096
            || (config.cache_mib > 0 && config.cache_latency == 0)
            || config.cache_latency > 1024
            || config.max_pending_per_sm == 0
            || config.max_pending_per_sm > 4096
        {
            return Err("invalid or unbounded memory configuration".into());
        }
        let cache = vec![
            [CacheEntry {
                tag: 0,
                stamp: 0,
                valid: false
            }; 4];
            config.cache_mib * 1024 * 1024 / 256
        ];
        let channels = (0..config.hbm_channels)
            .map(|_| Channel {
                waiting: VecDeque::new(),
                used: 0,
                free: 0,
                direction: None,
            })
            .collect();
        let pending = vec![0; config.sms];
        Ok(Self {
            config,
            now: 0,
            serial: 0,
            events: BinaryHeap::new(),
            transfers: HashMap::new(),
            channels,
            pending,
            pending_total: 0,
            line_use: HashMap::new(),
            merge: HashMap::new(),
            cache,
            stamp: 0,
            miss_wait: VecDeque::new(),
            mshrs: 0,
            cache_queue: VecDeque::new(),
            cache_tick: None,
            noc_queue: VecDeque::new(),
            noc_tick: None,
            deliveries: HashMap::new(),
            fanouts: HashMap::new(),
            acknowledgements: HashMap::new(),
            tickets: HashMap::new(),
            tokens: HashSet::new(),
            energy: Vec::new(),
            stats: Stats::default(),
        })
    }
    pub fn now(&self) -> u64 {
        self.now
    }
    pub fn stats(&self) -> &Stats {
        &self.stats
    }
    pub fn next_event_time(&self) -> Option<u64> {
        self.events.peek().map(|x| x.0.0)
    }
    pub fn pending(&self) -> usize {
        self.pending_total
    }
    pub fn drain_energy(&mut self) -> Vec<EnergyEvent> {
        std::mem::take(&mut self.energy)
    }
    /// Host tensor lifetime invalidation. Only legal at a fully acknowledged
    /// boundary; half-open byte range also invalidates partially covered lines.
    /// This is allocation bookkeeping, not a simulated program cache flush.
    pub fn invalidate_range(&mut self, start_byte: u64, end_byte: u64) -> Result<(), String> {
        if self.pending_total != 0 || !self.events.is_empty() {
            return Err("cache lifetime invalidation requires idle memory".into());
        }
        if start_byte > end_byte {
            return Err("invalid cache invalidation range".into());
        }
        if start_byte == end_byte {
            return Ok(());
        }
        let first = start_byte / 64;
        let last = (end_byte - 1) / 64;
        let line_count = last - first + 1;
        let sets = self.cache.len();
        if sets == 0 {
            return Ok(());
        }
        let invalidate = |set: &mut [CacheEntry; 4]| {
            for entry in set {
                if entry.valid && (first..=last).contains(&(entry.tag / 64)) {
                    entry.valid = false;
                }
            }
        };
        // A short contiguous range visits only its indexed sets. Once it covers
        // every set, scan the cache once; never iterate a potentially huge byte
        // range. Tags are still checked because each set also holds other lines.
        if line_count < sets as u64 {
            for line in first..=last {
                invalidate(&mut self.cache[(line % sets as u64) as usize]);
            }
        } else {
            for set in &mut self.cache {
                invalidate(set);
            }
        }
        Ok(())
    }
    fn id(&mut self) -> u64 {
        let x = self.serial;
        self.serial += 1;
        x
    }
    fn event(&mut self, time: u64, event: Event) {
        let order = self.id();
        self.events.push(Reverse((time, order, event)));
        self.stats.max_events = self.stats.max_events.max(self.events.len());
    }
    fn energy(&mut self, kind: EnergyKind, bytes: usize) {
        self.energy.push(EnergyEvent {
            cycle: self.now,
            kind,
            bytes: bytes as u64,
            duration: if matches!(kind, EnergyKind::HbmRead | EnergyKind::HbmWrite) {
                2
            } else {
                1
            },
            energy_pj: bytes as f64
                * match kind {
                    EnergyKind::HbmRead | EnergyKind::HbmWrite => 150.,
                    EnergyKind::HbmTurnaround => 0.,
                    EnergyKind::Noc => 1.,
                    EnergyKind::CacheRead | EnergyKind::CacheFill | EnergyKind::CacheInvalidate => {
                        2.5 + 0.15 * (self.config.cache_mib.max(1) as f64).log2()
                    }
                },
        });
    }
    /// Returns false only for finite frontend capacity or a same-line read/write
    /// ordering hazard. Invalid addresses/SM/byte counts return an error.
    pub fn try_submit(&mut self, request: Request) -> Result<bool, String> {
        if request.sm >= self.config.sms
            || !request.address.is_multiple_of(64)
            || request.valid_bytes == 0
            || request.valid_bytes > 64
        {
            return Err("invalid line request".into());
        }
        if self.tokens.contains(&request.token) {
            return Err("duplicate live memory token".into());
        }
        if self.pending[request.sm] >= self.config.max_pending_per_sm {
            return Ok(false);
        }
        if self
            .line_use
            .get(&request.address)
            .is_some_and(|u| u.writes > 0 || (request.write && u.reads > 0))
        {
            return Ok(false);
        }
        self.tokens.insert(request.token);
        self.pending[request.sm] += 1;
        self.pending_total += 1;
        self.stats.submitted += 1;
        self.stats.valid_bytes += request.valid_bytes as u64;
        self.stats.max_pending = self.stats.max_pending.max(self.pending_total);
        let u = self.line_use.entry(request.address).or_insert(LineUse {
            reads: 0,
            writes: 0,
        });
        if request.write {
            u.writes += 1;
        } else {
            u.reads += 1;
        }
        let id = self.id();
        self.transfers.insert(
            id,
            Transfer {
                request,
                destinations: vec![request],
            },
        );
        if request.write {
            self.invalidate(request.address);
        }
        // Every read address request pays eight upstream bytes; write requests
        // carry eight address bytes plus their full line. A write later returns an
        // eight-byte completion notification through the same shared network.
        self.enqueue_noc(Packet {
            id,
            sm: request.sm,
            remaining: if request.write { 72 } else { 8 },
            mode: 0,
        });
        Ok(true)
    }
    fn wake_cache(&mut self) {
        if self.cache_tick.is_none() {
            let t = self.now + 1;
            self.cache_tick = Some(t);
            self.event(t, Event::CacheTick);
        }
    }
    fn cache_index(&self, address: u64) -> usize {
        (address / 64) as usize % self.cache.len()
    }
    fn invalidate(&mut self, address: u64) {
        if self.cache.is_empty() {
            return;
        }
        let index = self.cache_index(address);
        for entry in &mut self.cache[index] {
            if entry.valid && entry.tag == address {
                entry.valid = false;
            }
        }
    }
    fn hit(&mut self, address: u64) -> bool {
        let index = self.cache_index(address);
        self.stamp += 1;
        for entry in &mut self.cache[index] {
            if entry.valid && entry.tag == address {
                entry.stamp = self.stamp;
                return true;
            }
        }
        false
    }
    fn fill(&mut self, address: u64) {
        if self.cache.is_empty() {
            return;
        }
        let index = self.cache_index(address);
        self.stamp += 1;
        let set = &mut self.cache[index];
        let slot = set
            .iter()
            .position(|x| x.valid && x.tag == address)
            .or_else(|| set.iter().position(|x| !x.valid))
            .unwrap_or_else(|| (0..4).min_by_key(|&i| set[i].stamp).unwrap());
        set[slot] = CacheEntry {
            tag: address,
            stamp: self.stamp,
            valid: true,
        };
    }
    fn miss(&mut self, id: u64) {
        let address = self.transfers[&id].request.address;
        if self.config.multicast || !self.cache.is_empty() {
            if let Some(&existing) = self
                .merge
                .get(&address)
                .filter(|id| self.transfers[*id].destinations.len() < 32)
            {
                let removed = self.transfers.remove(&id).unwrap();
                self.transfers
                    .get_mut(&existing)
                    .unwrap()
                    .destinations
                    .extend(removed.destinations);
                self.stats.merged_reads += 1;
                self.stats.miss_merged += 1;
                return;
            }
            self.merge.insert(address, id);
        }
        self.miss_wait.push_back(id);
        self.pump_misses();
    }
    fn pump_misses(&mut self) {
        while self.cache.is_empty() || self.mshrs < self.config.cache_mshrs {
            let Some(id) = self.miss_wait.pop_front() else {
                break;
            };
            if !self.cache.is_empty() {
                self.mshrs += 1;
                self.stats.max_cache_mshrs = self.stats.max_cache_mshrs.max(self.mshrs);
            }
            self.enqueue_hbm(id);
        }
    }
    fn enqueue_hbm(&mut self, id: u64) {
        let channel = (self.transfers[&id].request.address / 256) as usize % self.channels.len();
        self.channels[channel].waiting.push_back(id);
        self.pump_channel(channel);
    }
    fn pump_channel(&mut self, index: usize) {
        while self.channels[index].used < self.config.hbm_queue_depth {
            let Some(id) = self.channels[index].waiting.pop_front() else {
                break;
            };
            self.channels[index].used += 1;
            self.stats.max_hbm_inflight = self
                .stats
                .max_hbm_inflight
                .max(self.channels.iter().map(|c| c.used).sum());
            self.event(self.now + 250, Event::HbmReady(id));
        }
    }
    fn enqueue_noc(&mut self, packet: Packet) {
        self.noc_queue.push_back(packet);
        if self.noc_tick.is_none() {
            let t = self.now + 1;
            self.noc_tick = Some(t);
            self.event(t, Event::NocTick);
        }
    }
    fn delivery(&mut self, request: Request, ticket: Option<u64>) {
        let id = self.id();
        self.deliveries.insert(id, (request, ticket));
        self.enqueue_noc(Packet {
            id,
            sm: request.sm,
            remaining: if request.write { 8 } else { 64 },
            mode: 0,
        });
    }
    fn finish(&mut self, request: Request, ticket: Option<u64>, completed: &mut Vec<Completion>) {
        self.acknowledgements
            .insert(request.token, (request, ticket));
        completed.push(Completion {
            request,
            cycle: self.now,
        });
    }
    /// Return credits only after the scheduler has finished local RF/SH receive.
    /// A multicast transaction retains its channel slot until its last fanout ack.
    pub fn ack_local(&mut self, token: u64) -> Result<(), String> {
        let (request, ticket) = self
            .acknowledgements
            .remove(&token)
            .ok_or("unknown or premature local acknowledgement")?;
        self.tokens.remove(&token);
        self.pending[request.sm] -= 1;
        self.pending_total -= 1;
        self.stats.completed += 1;
        let usage = self.line_use.get_mut(&request.address).unwrap();
        if request.write {
            usage.writes -= 1;
        } else {
            usage.reads -= 1;
        }
        if usage.reads == 0 && usage.writes == 0 {
            self.line_use.remove(&request.address);
        }
        if let Some(id) = ticket {
            let (channel, left) = self.tickets.get_mut(&id).unwrap();
            *left -= 1;
            if *left == 0 {
                let channel = *channel;
                self.tickets.remove(&id);
                self.channels[channel].used -= 1;
                self.pump_channel(channel);
            }
        }
        Ok(())
    }
    fn return_read(&mut self, id: u64) {
        let transfer = self.transfers.remove(&id).unwrap();
        let request = transfer.request;
        let channel = (request.address / 256) as usize % self.channels.len();
        self.tickets
            .insert(id, (channel, transfer.destinations.len()));
        if self.merge.get(&request.address) == Some(&id) {
            self.merge.remove(&request.address);
        }
        if !self.cache.is_empty() {
            self.fill(request.address);
            self.mshrs -= 1;
        }
        if self.config.multicast && transfer.destinations.len() > 1 {
            self.stats.multicast_destinations += transfer.destinations.len() as u64;
            let packet_id = self.id();
            self.fanouts.insert(
                packet_id,
                transfer
                    .destinations
                    .into_iter()
                    .map(|r| (r, Some(id)))
                    .collect(),
            );
            self.enqueue_noc(Packet {
                id: packet_id,
                sm: 0,
                remaining: 64,
                mode: 1,
            });
        } else {
            for destination in transfer.destinations {
                self.delivery(destination, Some(id));
            }
        }
        self.pump_misses();
    }
    /// Advances through events without per-cycle idle work. Panics on backward
    /// time, which is a scheduler programming error, not a student input error.
    pub fn advance_to(&mut self, target: u64) -> Vec<Completion> {
        assert!(target >= self.now, "memory time cannot go backwards");
        let mut completed = Vec::new();
        while self.next_event_time().is_some_and(|t| t <= target) {
            let Reverse((time, _, event)) = self.events.pop().unwrap();
            self.now = time;
            match event {
                Event::CacheTick => {
                    self.cache_tick = None;
                    let mut credit = self.config.cache_bytes_per_cycle;
                    while credit > 0 {
                        let Some((id, mut remaining, fill)) = self.cache_queue.pop_front() else {
                            break;
                        };
                        let n = remaining.min(credit);
                        remaining -= n;
                        credit -= n;
                        self.energy(
                            match fill {
                                1 => EnergyKind::CacheFill,
                                2 => EnergyKind::CacheInvalidate,
                                _ => EnergyKind::CacheRead,
                            },
                            n,
                        );
                        if remaining == 0 {
                            self.event(
                                self.now + self.config.cache_latency,
                                match fill {
                                    1 => Event::Fill(id),
                                    2 => Event::WriteLookup(id),
                                    _ => Event::Lookup(id),
                                },
                            );
                        } else {
                            self.cache_queue.push_front((id, remaining, fill));
                        }
                    }
                    if !self.cache_queue.is_empty() {
                        self.wake_cache();
                    }
                }
                Event::Lookup(id) => {
                    let address = self.transfers[&id].request.address;
                    if self.hit(address) {
                        self.stats.cache_hits += 1;
                        let transfer = self.transfers.remove(&id).unwrap();
                        for destination in transfer.destinations {
                            self.delivery(destination, None);
                        }
                    } else {
                        self.stats.cache_misses += 1;
                        self.miss(id);
                    }
                }
                Event::NocTick => {
                    self.noc_tick = None;
                    let mut global = self.config.noc_bytes_per_cycle;
                    let mut endpoints = [self.config.sm_noc_bytes_per_cycle; 32];
                    let count = self.noc_queue.len();
                    for _ in 0..count {
                        let mut packet = self.noc_queue.pop_front().unwrap();
                        let n = packet
                            .remaining
                            .min(if packet.mode == 2 { usize::MAX } else { global })
                            .min(if packet.mode == 1 {
                                usize::MAX
                            } else {
                                endpoints[packet.sm]
                            });
                        packet.remaining -= n;
                        let trunk = if packet.mode == 2 { 0 } else { n };
                        let endpoint = if packet.mode == 1 { 0 } else { n };
                        global -= trunk;
                        endpoints[packet.sm] -= endpoint;
                        self.stats.noc_global_bytes += trunk as u64;
                        self.stats.noc_endpoint_bytes += endpoint as u64;
                        self.stats.noc_bytes += (trunk + endpoint) as u64;
                        if n > 0 {
                            self.energy.push(EnergyEvent {
                                cycle: self.now,
                                kind: EnergyKind::Noc,
                                bytes: (trunk + endpoint) as u64,
                                duration: 1,
                                energy_pj: (trunk + endpoint) as f64,
                            });
                        }
                        if packet.remaining == 0 {
                            self.event(
                                self.now + if packet.mode == 1 { 0 } else { 6 },
                                if packet.mode == 1 {
                                    Event::Fanout(packet.id)
                                } else {
                                    Event::Arrive(packet.id)
                                },
                            );
                        } else {
                            self.noc_queue.push_back(packet);
                        }
                    }
                    if !self.noc_queue.is_empty() {
                        let t = self.now + 1;
                        self.noc_tick = Some(t);
                        self.event(t, Event::NocTick);
                    }
                }
                Event::Fanout(id) => {
                    let destinations = self.fanouts.remove(&id).unwrap();
                    for (request, ticket) in destinations {
                        let delivery = self.id();
                        self.deliveries.insert(delivery, (request, ticket));
                        self.enqueue_noc(Packet {
                            id: delivery,
                            sm: request.sm,
                            remaining: 64,
                            mode: 2,
                        });
                    }
                }
                Event::Arrive(id) => {
                    if let Some((request, ticket)) = self.deliveries.remove(&id) {
                        self.finish(request, ticket, &mut completed);
                    } else {
                        let write = self.transfers[&id].request.write;
                        if self.cache.is_empty() {
                            if write {
                                self.enqueue_hbm(id);
                            } else {
                                self.miss(id);
                            }
                        } else {
                            self.cache_queue
                                .push_back((id, 64, if write { 2 } else { 0 }));
                            self.wake_cache();
                        }
                    }
                }
                Event::WriteLookup(id) => self.enqueue_hbm(id),
                Event::HbmReady(id) => {
                    let request = self.transfers[&id].request;
                    let index = (request.address / 256) as usize % self.channels.len();
                    let channel = &mut self.channels[index];
                    let switch = channel.direction.is_some_and(|w| w != request.write);
                    let start = self.now.max(channel.free) + if switch { 4 } else { 0 };
                    channel.free = start + 2;
                    channel.direction = Some(request.write);
                    self.stats.direction_switches += u64::from(switch);
                    if switch {
                        self.event(start - 4, Event::Turnaround);
                    }
                    self.event(start, Event::HbmStart(id));
                    self.event(start + 2, Event::HbmDone(id));
                }
                Event::Turnaround => self.energy.push(EnergyEvent {
                    cycle: self.now,
                    kind: EnergyKind::HbmTurnaround,
                    bytes: 0,
                    duration: 4,
                    energy_pj: 200.,
                }),
                Event::HbmStart(id) => {
                    let write = self.transfers[&id].request.write;
                    self.energy(
                        if write {
                            EnergyKind::HbmWrite
                        } else {
                            EnergyKind::HbmRead
                        },
                        64,
                    );
                }
                Event::Fill(id) => self.return_read(id),
                Event::HbmDone(id) => {
                    let request = self.transfers[&id].request;
                    if request.write {
                        self.stats.hbm_writes += 1;
                        self.transfers.remove(&id);
                        let channel = (request.address / 256) as usize % self.channels.len();
                        self.tickets.insert(id, (channel, 1));
                        self.delivery(request, Some(id));
                    } else {
                        self.stats.hbm_reads += 1;
                        if self.cache.is_empty() {
                            self.return_read(id);
                        } else {
                            self.cache_queue.push_back((id, 64, 1));
                            self.wake_cache();
                        }
                    }
                }
            }
        }
        self.now = target;
        completed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(sm: usize, address: u64) -> Request {
        Request {
            sm,
            group: sm as u64,
            token: address + sm as u64,
            address,
            valid_bytes: 64,
            write: false,
        }
    }
    fn drain(memory: &mut Memory) -> Vec<Completion> {
        let mut out = vec![];
        while let Some(t) = memory.next_event_time() {
            let completed = memory.advance_to(t);
            for c in &completed {
                memory.ack_local(c.request.token).unwrap();
            }
            out.extend(completed);
            memory.drain_energy();
        }
        assert_eq!(memory.pending(), 0);
        out
    }
    #[test]
    fn hit_eviction_and_write_invalidation() {
        let mut m = Memory::new(MemoryConfig {
            cache_mib: 1,
            ..Default::default()
        })
        .unwrap();
        m.try_submit(request(0, 0)).unwrap();
        let first = drain(&mut m)[0].cycle;
        let start = m.now();
        m.try_submit(request(0, 0)).unwrap();
        assert!(drain(&mut m)[0].cycle - start < first);
        assert_eq!(m.stats.cache_hits, 1);
        for i in 1..=4 {
            m.try_submit(request(0, i * 262144)).unwrap();
            drain(&mut m);
        }
        m.try_submit(request(0, 0)).unwrap();
        drain(&mut m);
        assert_eq!(m.stats.cache_hits, 1);
        let mut write = request(0, 0);
        write.write = true;
        m.try_submit(write).unwrap();
        assert!(!m.try_submit(request(1, 0)).unwrap());
        drain(&mut m);
        m.try_submit(request(0, 0)).unwrap();
        drain(&mut m);
        assert_eq!(m.stats.cache_hits, 1);
        assert_eq!(m.stats.hbm_writes, 1);
    }
    #[test]
    fn multicast_pays_each_destination_and_no_merge_when_disabled() {
        for enabled in [false, true] {
            let mut m = Memory::new(MemoryConfig {
                multicast: enabled,
                ..Default::default()
            })
            .unwrap();
            for sm in 0..4 {
                assert!(m.try_submit(request(sm, 0)).unwrap());
            }
            assert_eq!(drain(&mut m).len(), 4);
            assert_eq!(m.stats.hbm_reads, if enabled { 1 } else { 4 });
            assert_eq!(m.stats.noc_global_bytes, if enabled { 96 } else { 288 });
            assert_eq!(m.stats.noc_endpoint_bytes, 288);
        }
    }
    #[test]
    fn bounded_frontend_hbm_mshrs_and_progress() {
        let mut m = Memory::new(MemoryConfig {
            hbm_channels: 1,
            hbm_queue_depth: 1,
            cache_mib: 1,
            cache_mshrs: 1,
            max_pending_per_sm: 2,
            ..Default::default()
        })
        .unwrap();
        for sm in 0..4 {
            for n in 0..2 {
                assert!(
                    m.try_submit(request(sm, ((sm * 2 + n) * 64) as u64))
                        .unwrap()
                );
            }
            assert!(!m.try_submit(request(sm, 4096)).unwrap());
        }
        assert_eq!(drain(&mut m).len(), 8);
        assert_eq!(m.stats.max_hbm_inflight, 1);
        assert_eq!(m.stats.max_cache_mshrs, 1);
        assert_eq!(m.stats.max_pending, 8);
    }
    #[test]
    fn noc_bandwidth_and_direction_turnaround_are_paid() {
        let run = |bw| {
            let mut m = Memory::new(MemoryConfig {
                hbm_channels: 1,
                noc_bytes_per_cycle: bw,
                ..Default::default()
            })
            .unwrap();
            for sm in 0..4 {
                m.try_submit(request(sm, (sm * 64) as u64)).unwrap();
            }
            drain(&mut m).iter().map(|x| x.cycle).max().unwrap()
        };
        assert!(run(8) > run(256));
        let mut m = Memory::new(MemoryConfig {
            hbm_channels: 1,
            ..Default::default()
        })
        .unwrap();
        m.try_submit(request(0, 0)).unwrap();
        let mut w = request(1, 64);
        w.write = true;
        m.try_submit(w).unwrap();
        drain(&mut m);
        assert_eq!(m.stats.direction_switches, 1);
    }
    #[test]
    fn read_inflight_prevents_stale_write_fill() {
        let mut m = Memory::new(MemoryConfig {
            cache_mib: 1,
            multicast: true,
            ..Default::default()
        })
        .unwrap();
        m.try_submit(request(0, 0)).unwrap();
        let mut w = request(1, 0);
        w.write = true;
        assert!(!m.try_submit(w).unwrap());
        drain(&mut m);
        assert!(m.try_submit(w).unwrap());
        drain(&mut m);
        m.try_submit(request(0, 0)).unwrap();
        drain(&mut m);
        assert_eq!(m.stats.cache_hits, 0);
    }
    #[test]
    fn credit_waits_for_local_ack_and_energy_never_arrives_from_future() {
        let mut m = Memory::new(MemoryConfig {
            hbm_channels: 1,
            hbm_queue_depth: 1,
            max_pending_per_sm: 1,
            ..Default::default()
        })
        .unwrap();
        m.try_submit(request(0, 0)).unwrap();
        m.try_submit(request(1, 64)).unwrap();
        let mut first = None;
        while let Some(t) = m.next_event_time() {
            let done = m.advance_to(t);
            for e in m.drain_energy() {
                assert!(e.cycle <= t);
            }
            if !done.is_empty() {
                first = Some(done[0]);
                break;
            }
        }
        assert_eq!(m.stats.hbm_reads, 1);
        assert_eq!(m.pending(), 2);
        assert!(!m.try_submit(request(0, 128)).unwrap());
        assert!(m.next_event_time().is_none());
        let c = first.unwrap();
        m.ack_local(c.request.token).unwrap();
        assert!(m.ack_local(c.request.token).is_err());
        assert_eq!(drain(&mut m).len(), 1);
        assert_eq!(m.stats.hbm_reads, 2);
    }
    #[test]
    fn idle_address_reuse_invalidates_persistent_cache() {
        let mut m = Memory::new(MemoryConfig {
            cache_mib: 1,
            ..Default::default()
        })
        .unwrap();
        m.try_submit(request(0, 0)).unwrap();
        assert!(m.invalidate_range(0, 64).is_err());
        drain(&mut m);
        m.invalidate_range(1, 2).unwrap();
        m.try_submit(request(0, 0)).unwrap();
        drain(&mut m);
        assert_eq!(m.stats.hbm_reads, 2);
        assert_eq!(m.stats.cache_hits, 0);
    }
    #[test]
    fn multicast_fanout_is_bounded_and_all_destinations_keep_credit() {
        let mut m = Memory::new(MemoryConfig {
            multicast: true,
            max_pending_per_sm: 64,
            ..Default::default()
        })
        .unwrap();
        for token in 0..64 {
            let mut r = request(0, 0);
            r.token = token;
            m.try_submit(r).unwrap();
        }
        assert_eq!(drain(&mut m).len(), 64);
        assert_eq!(m.stats.hbm_reads, 2);
        assert_eq!(m.stats.noc_global_bytes, 64 * 8 + 2 * 64);
        assert_eq!(m.stats.noc_endpoint_bytes, 64 * (8 + 64));
    }
    #[test]
    fn isolated_latency_matches_published_pipeline() {
        for cache_mib in [0, 1] {
            let mut m = Memory::new(MemoryConfig {
                cache_mib,
                ..Default::default()
            })
            .unwrap();
            m.try_submit(request(0, 0)).unwrap();
            assert_eq!(
                drain(&mut m)[0].cycle,
                if cache_mib == 0 { 266 } else { 308 }
            );
            if cache_mib != 0 {
                let start = m.now();
                m.try_submit(request(0, 0)).unwrap();
                assert_eq!(drain(&mut m)[0].cycle - start, 35);
            }
        }
    }
    #[test]
    fn stepping_each_cycle_agrees_with_jumping_events() {
        let run = |single_cycle: bool| {
            let mut m = Memory::new(MemoryConfig {
                cache_mib: 1,
                multicast: true,
                noc_bytes_per_cycle: 16,
                ..Default::default()
            })
            .unwrap();
            for i in 0..32 {
                let mut r = request(i % 4, (i % 7 * 64) as u64);
                r.token = i as u64;
                m.try_submit(r).unwrap();
            }
            let mut out = vec![];
            let mut energy = vec![];
            while let Some(next) = m.next_event_time() {
                let t = if single_cycle { m.now() + 1 } else { next };
                let done = m.advance_to(t);
                for c in &done {
                    m.ack_local(c.request.token).unwrap();
                }
                out.extend(done);
                energy.extend(
                    m.drain_energy()
                        .into_iter()
                        .map(|e| (e.cycle, e.duration, e.energy_pj.to_bits())),
                );
            }
            (out, energy)
        };
        assert_eq!(run(false), run(true));
    }
    #[test]
    fn cache_native_miss_merge_does_not_require_network_multicast() {
        for multicast in [false, true] {
            let mut m = Memory::new(MemoryConfig {
                cache_mib: 1,
                multicast,
                ..Default::default()
            })
            .unwrap();
            for sm in 0..4 {
                m.try_submit(request(sm, 0)).unwrap();
            }
            assert_eq!(drain(&mut m).len(), 4);
            assert_eq!(m.stats.hbm_reads, 1);
            assert_eq!(m.stats.miss_merged, 3);
            assert_eq!(
                m.stats.noc_global_bytes,
                32 + if multicast { 64 } else { 256 }
            );
            assert_eq!(m.stats.noc_endpoint_bytes, 32 + 256);
            assert_eq!(
                m.stats.multicast_destinations,
                if multicast { 4 } else { 0 }
            );
        }
    }
    #[test]
    fn multicast_saves_congested_trunk_but_pays_every_endpoint() {
        let run = |multicast| {
            let mut m = Memory::new(MemoryConfig {
                cache_mib: 1,
                multicast,
                noc_bytes_per_cycle: 8,
                ..Default::default()
            })
            .unwrap();
            for sm in 0..4 {
                m.try_submit(request(sm, 0)).unwrap();
            }
            let cycles = drain(&mut m).iter().map(|c| c.cycle).max().unwrap();
            (cycles, m.stats.noc_endpoint_bytes)
        };
        let plain = run(false);
        let broadcast = run(true);
        assert!(broadcast.0 < plain.0);
        assert_eq!(broadcast.1, plain.1);
    }
    #[test]
    fn returning_broadcast_cannot_retroactively_absorb_late_reader() {
        let mut m = Memory::new(MemoryConfig {
            multicast: true,
            noc_bytes_per_cycle: 8,
            ..Default::default()
        })
        .unwrap();
        for sm in 0..2 {
            m.try_submit(request(sm, 0)).unwrap();
        }
        while m.stats.hbm_reads == 0 {
            let t = m.next_event_time().unwrap();
            assert!(m.advance_to(t).is_empty());
            m.drain_energy();
        }
        m.try_submit(request(2, 0)).unwrap();
        assert_eq!(drain(&mut m).len(), 3);
        assert_eq!(m.stats.hbm_reads, 2);
        assert_eq!(m.stats.multicast_destinations, 2);
    }
    #[test]
    fn disabled_cache_accepts_zero_derived_resources() {
        let mut m = Memory::new(MemoryConfig {
            cache_mib: 0,
            cache_mshrs: 0,
            cache_bytes_per_cycle: 0,
            cache_latency: 0,
            max_pending_per_sm: 128,
            ..Default::default()
        })
        .unwrap();
        m.try_submit(request(0, 0)).unwrap();
        assert_eq!(drain(&mut m).len(), 1);
        assert!(
            Memory::new(MemoryConfig {
                cache_mib: 1,
                cache_mshrs: 0,
                ..Default::default()
            })
            .is_err()
        );
    }
    #[test]
    fn write_pays_address_payload_and_completion_notification() {
        for cache_mib in [0, 1] {
            let mut m = Memory::new(MemoryConfig {
                cache_mib,
                sm_noc_bytes_per_cycle: 64,
                noc_bytes_per_cycle: 256,
                ..Default::default()
            })
            .unwrap();
            let mut write = request(0, 0);
            write.write = true;
            m.try_submit(write).unwrap();
            let mut completion = None;
            let mut energy = 0.;
            while let Some(t) = m.next_event_time() {
                let done = m.advance_to(t);
                energy += m
                    .drain_energy()
                    .into_iter()
                    .map(|e| e.energy_pj)
                    .sum::<f64>();
                if m.stats.hbm_writes == 1 && done.is_empty() {
                    assert!(m.ack_local(write.token).is_err());
                    assert_eq!(m.pending(), 1);
                }
                for c in done {
                    completion = Some(c);
                    m.ack_local(c.request.token).unwrap();
                }
            }
            assert_eq!(
                completion.unwrap().cycle,
                if cache_mib == 0 { 267 } else { 288 }
            );
            assert_eq!(m.stats.noc_global_bytes, 72 + 8);
            assert_eq!(m.stats.noc_endpoint_bytes, 72 + 8);
            assert_eq!(m.stats.noc_bytes, 160);
            assert_eq!(m.stats.hbm_writes, 1);
            assert_eq!(
                energy,
                9600. + 160. + if cache_mib == 0 { 0. } else { 160. }
            );
        }
    }
    #[test]
    fn indexed_lifetime_invalidation_matches_full_tag_scan() {
        let mut m = Memory::new(MemoryConfig {
            cache_mib: 1,
            ..Default::default()
        })
        .unwrap();
        let sets = m.cache.len();
        for (index, set) in m.cache.iter_mut().enumerate() {
            for (way, entry) in set.iter_mut().enumerate() {
                *entry = CacheEntry {
                    tag: ((index + way * sets) * 64) as u64,
                    stamp: (way * sets + index) as u64,
                    valid: (way + index) % 3 != 0,
                };
            }
        }
        let original = m.cache.clone();
        let mut ranges = vec![
            (0, 0),
            (1, 2),
            (63, 64),
            (63, 65),
            (0, u64::MAX),
            (u64::MAX - 64, u64::MAX),
            ((sets * 64 - 1) as u64, (sets * 64 + 65) as u64),
            (0, sets as u64 * 64),
            (0, sets as u64 * 64 - 1),
        ];
        for i in 0..32u64 {
            let start = (i * 7919) % (sets as u64 * 256);
            ranges.push((start, start + (i * 1231) % 32768 + 1));
        }
        for (start, end) in ranges {
            let mut expected = original.clone();
            if start != end {
                for set in &mut expected {
                    for entry in set {
                        if entry.valid && (start / 64..=(end - 1) / 64).contains(&(entry.tag / 64))
                        {
                            entry.valid = false;
                        }
                    }
                }
            }
            m.cache.clone_from(&original);
            m.invalidate_range(start, end).unwrap();
            assert_eq!(m.cache, expected, "range {start}..{end}");
        }
    }
}
