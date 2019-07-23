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
#[derive(Debug)]
struct Neuron {
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

    // Property
    // sum(pneg) + sum(ppos) + pdissipate == 1
    // where pdissipate is the probability that the signal dissipates
}

impl Neuron {
    /// Receives input.
    fn receive(&mut self, value: i16) {
        self.reserve += value;
        if self.reserve < 0 {
            self.reserve = 0;
        }
    }

    /// Excites neuron and possibly returns action potential signal.
    /// A downstream neuron will not receive both negative and 
    /// positive signals simultaneously.
    fn excite(&self) -> Option<(Signal, Signal)> {
        if self.reserve > 0 {
            let mut ntargets = Vec::new();
            let mut ptargets = Vec::new();

            use rand_distr::{Uniform, Distribution};
            let mut rng = rand::thread_rng();
            let unif = Uniform::new(0.0, 1.0);

            // propagate signal to each target
            for i in 0..self.targets.len() {
                let target = self.targets[i];
                let pneg = self.pnegatives[i];
                let ppos = self.ppositives[i];

                let r: f32 = unif.sample(&mut rng);
                if r < pneg {
                    ntargets.push(target);
                } else if r < ppos {
                    ptargets.push(target);
                } else {
                    // output dissipates
                }
            }
            
            Some((
                Signal { value: -1, rate: 0.0, targets: ntargets },
                Signal { value:  1, rate: 0.0, targets: ptargets },
            ))
        } else {
            // neuron cannot fire yet
            None
        }
    }
}

/// External signal
#[derive(Debug)]
struct Signal {
    /// signal value
    value: i16,
    /// arrival rates
    rate: f32,
    /// index of target neurons
    targets: Vec<usize>,
}

/// Random neuron network
#[derive(Debug)]
struct Network {
    /// positive and negative signals
    signals: Vec<Signal>,
    /// neurons
    neurons: Vec<Neuron>,
    /// priority queue tracking next node to fire
    queue: Queue
}

impl Network {
    // Creates a network.
    fn new(signals: Vec<Signal>, neurons: Vec<Neuron>) -> Self {
        let queue = Queue::new(&signals, &neurons);
        Network {
            signals: signals,
            neurons: neurons,
            queue: queue
        }
    }

    // Updates network by firing the next node.
    fn update(&mut self) {
        // get the next node to fire
        // we use unwrap because priority queue must always be populated
        let node = self.queue.pop().unwrap();

        let rate;
        if node.index < self.signals.len() {
            // a signal arrives
            let signal = &self.signals[node.index];
            rate = signal.rate;
            for i in &signal.targets {
                self.neurons[*i].receive(signal.value);
            }
        } else {
            // a neuron fires
            let index = node.index - self.signals.len();
            let neuron = &self.neurons[index];
            rate = neuron.rate;
            match neuron.excite() {
                Some( (pos, neg) ) => {
                    // after neuron's activation, reduce its reserve
                    self.neurons[index].receive(-1);
                    // send positive signals
                    for i in pos.targets {
                        self.neurons[i].receive(pos.value);
                    }
                    // send negative signals
                    for i in neg.targets {
                        self.neurons[i].receive(neg.value);
                    }
                },
                _ => (),
            }
        }

        // push node back onto the queue for next firing
        self.queue.push(
            Node{ wait: rexp(rate, &mut rand::thread_rng()), index: node.index }
        );
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
            },
            Neuron {
                reserve: 0,
                rate: 2.0,
                targets: vec![2, 3],
                pnegatives: vec![0.0, 0.5],
                ppositives: vec![0.5, 0.0],
            },
            Neuron {
                reserve: 0,
                rate: 1.1,
                targets: vec![3],
                pnegatives: vec![0.0],
                ppositives: vec![1.0],
            },
            Neuron {
                reserve: 0,
                rate: r,
                targets: Vec::new(),
                pnegatives: Vec::new(),
                ppositives: Vec::new(),
            },
        ];

        let signals = vec![
            Signal { value: 1, rate: 3.0, targets: vec![0] },
            Signal { value: 1, rate: 3.0, targets: vec![1] },
        ];

        let mut net = Network::new(signals, neurons);
        for i in 0..100 {
            net.update();
        }
        println!("{:?}", net.neurons[3]);
    }
}
