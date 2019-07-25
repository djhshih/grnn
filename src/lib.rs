extern crate rand;
extern crate rand_distr;

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Sample from exponential distribution
fn rexp<R: rand::Rng>(rate: f32, mut rng: &mut R) -> f32 {
    use rand_distr::{Exp, Distribution};
    if rate > 0.0 {
        let exp = Exp::new(rate).unwrap();
        exp.sample(&mut rng)
    } else {
        std::f32::INFINITY
    }
}

/// Stochastically firing neuron
#[derive(Clone, Debug)]
pub struct Neuron {
    /// reserve potential
    reserve: i16,
    /// firing rate >= 0
    rate: f32,
    /// index of target neurons
    targets: Vec<usize>,
    /// probability of outgoing negative signal
    pnegatives: Vec<f32>,
    /// probability of outgoing positive signal
    ppositives: Vec<f32>,
    /// whether neuron is being record
    record: bool,

    // Property
    // sum(pneg) + sum(ppos) + pdissipate == 1
    // where pdissipate is the probability that the signal dissipates
}

impl Neuron {
    /// Receives input.
    pub fn receive(&mut self, value: i16) {
        self.reserve += value;
        if self.reserve < 0 {
            self.reserve = 0;
        }
    }

    /// Excites neuron.
    /// Requires index of neuron to record the source.
    /// At most one action potential event is returned.
    pub fn excite(&self, index: usize) -> Option<Event> {
        if self.reserve > 0 {

            use rand_distr::{Uniform, Distribution};
            let mut rng = rand::thread_rng();
            let unif = Uniform::new(0.0, 1.0);
            let r: f32 = unif.sample(&mut rng);

            // propagate signal to a target
            // target uses 1-index and 0 is a placeholder for NA
            let mut target = 0;
            let mut signal_value = 0;
            // assume that sum(pneg) + sum(ppos) + pdissipate == 1
            let mut cumsum: f32 = 0.0;
            for i in 0..self.targets.len() {
                let t = self.targets[i];
                let pneg = self.pnegatives[i];
                let ppos = self.ppositives[i];

                cumsum += pneg;
                if r <= cumsum {
                    target = t + 1;
                    signal_value = -1;
                    break;
                }

                cumsum += ppos;
                if r <= cumsum {
                    target = t + 1;
                    signal_value = 1;
                    break;
                }
            }

            if target > 0 {
                // convert target index from 1-based to 0-based
                Some( Event { value: signal_value, source: index, target: target - 1 } )
            } else {
                // output signal dissipates
                Some( Event { value: 0, source: index, target: 0 } )
            }
        } else {
            // neuron cannot fire yet
            None
        }
    }
}

/// External signal
#[derive(Debug)]
pub struct Signal {
    /// signal value
    value: i16,
    /// arrival rates
    rate: f32,
    /// index of target neurons
    targets: Vec<usize>,
}

/// Action potential event
#[derive(Debug)]
pub struct Event {
    /// signal value
    value: i16,
    /// index of source neuron
    source: usize,
    /// index of target neuron
    target: usize,
}

/// Random neuron network
#[derive(Debug)]
pub struct Network {
    /// positive and negative signals
    signals: Vec<Signal>,
    /// neurons
    neurons: Vec<Neuron>,
    /// priority queue tracking next node to fire
    queue: Queue
}

impl Network {
    // Creates a network.
    pub fn new(signals: Vec<Signal>, neurons: Vec<Neuron>) -> Self {
        let queue = Queue::new(&signals, &neurons);
        Network {
            signals: signals,
            neurons: neurons,
            queue: queue
        }
    }

    pub fn len(&self) -> usize {
        self.neurons.len()
    }

    /// Runs the network until the next neuron under recording fires.
    /// Returns the elapsed time and the event.
    pub fn run(&mut self) -> (f64, Event) {
        let mut t = 0.0;
        let event;
        loop {
            let (dt, record, opt) = self.update();
            t += dt;
            if record {
                match opt {
                    Some(ev) => {
                        event = ev;
                        break;
                    },
                    _ => ()
                }
            }
        }
        (t, event) 
    }

    /// Updates the network by firing the next node.
    /// Returns the elapsed time,
    /// whether the fired neuron is being recorded,
    /// and the action potential event (if any)
    pub fn update(&mut self) -> (f64, bool, Option<Event>) {
        // get the next node to fire
        // we use unwrap because priority queue must always be populated
        let node = self.queue.pop().unwrap();

        let mut record = false;
        let mut option = None;
        let rate;
        if node.index < self.signals.len() {
            // a signal arrives
            let signal = &self.signals[node.index];
            rate = signal.rate;
            for i in &signal.targets {
                self.neurons[*i].receive(signal.value);
            }
        } else {
            let index = node.index - self.signals.len();
            let neuron = &self.neurons[index];
            rate = neuron.rate;
            match neuron.excite(index) {
                Some(ev) => {
                    // a neuron fires
                    if neuron.record {
                        record = true;
                    }
                    // after neuron's activation, reduce its reserve
                    self.neurons[index].receive(-1);
                    // send signal to postsynaptic neuron
                    if ev.value != 0 {
                        self.neurons[ev.target].receive(ev.value);
                    }
                    option = Some(ev);
                },
                _ => ()
            }
        }

        // push node back onto the queue for next firing
        self.queue.push(
            Node{ wait: rexp(rate, &mut rand::thread_rng()), index: node.index }
        );

        (node.wait as f64, record, option)
    }
}

/// Placeholder for a signal or a neuron
#[derive(Debug)]
struct Node {
    /// wait time until next firing
    wait: f32,
    /// index of node;
    /// order of index of nodes: signals, neurons
    index: usize,
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.wait == other.wait && self.index == other.index
    }
}

impl Eq for Node {}

// prioritize least waiting time
impl Ord for Node {
    fn cmp(&self, other: &Node) -> Ordering {
        // ordering is flipped: other.wait < self.wait
        // manual implementation requried since cmp is not implemented for f32
        let ord;
        if other.wait < self.wait {
            ord = Ordering::Less;
        } else if other.wait > self.wait {
            ord = Ordering::Greater;
        } else {
            ord = Ordering::Equal;
        }
        
        // break tie by index to ensure order stability
        ord.then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Node) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Priority queue
/// Determines which node to fire next.
#[derive(Debug)]
struct Queue {
    queue: BinaryHeap<Node>
}

impl Queue {
    /// Creates a priority queue for a network.
    fn new(signals: &[Signal], neurons: &[Neuron]) -> Queue {
        let mut queue = BinaryHeap::new();
        let mut rng = rand::thread_rng();
        let mut index: usize = 0;

        for node in signals {
            queue.push(
                Node{ wait: rexp(node.rate, &mut rng), index: index }
            );
            index += 1;
        }

        for node in neurons {
            queue.push(
                Node{ wait: rexp(node.rate, &mut rng), index: index }
            );
            index += 1;
        }

        Queue{ queue: queue }
    }

    /// Pushes a node onto the queue while maintaining sorted order.
    fn push(&mut self, node: Node) {
        self.queue.push(node)
    }

    /// Removes next node and return it, or None if queue is empty.
    fn pop(&mut self) -> Option<Node> {
        self.queue.pop()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xor_rnn() {
        let r = 0.1;
        let neurons = vec![
            Neuron {
                reserve: 0,
                rate: 2.0,
                targets: vec![2, 3],
                pnegatives: vec![0.0, 0.5],
                ppositives: vec![0.5, 0.0],
                record: false,
            },
            Neuron {
                reserve: 0,
                rate: 2.0,
                targets: vec![2, 3],
                pnegatives: vec![0.0, 0.5],
                ppositives: vec![0.5, 0.0],
                record: false,
            },
            Neuron {
                reserve: 0,
                rate: 1.1,
                targets: vec![3],
                pnegatives: vec![0.0],
                ppositives: vec![1.0],
                record: false,
            },
            Neuron {
                reserve: 0,
                rate: r,
                targets: Vec::new(),
                pnegatives: Vec::new(),
                ppositives: Vec::new(),
                record: true,
            },
        ];

        {
            let signals = vec![
                Signal { value: 1, rate: 2.0, targets: vec![0] },
                Signal { value: 1, rate: 2.0, targets: vec![1] },
            ];

            let mut net = Network::new(signals, neurons);
            let mut t = 0.0;
            let mut count = 0;
            let mut steps = 0;
            let mut spike_counts = vec![0; net.len()];
            while t < 10.0 {
                let (dt, record, opt) = net.update();
                match opt {
                    Some(event) => {
                        spike_counts[event.source] += 1;
                        println!("{}: {} -> {}", event.value, event.source, event.target);
                    },
                    _ => ()
                }
                if record {
                    count += 1;
                }
                t += dt;
                steps += 1;
            }

            println!("{:?}", net.neurons);
            println!("{:?}", spike_counts);

            println!("steps: {}, count: {}", steps, count);
            assert!(count <= 5);
        }
    }
}
