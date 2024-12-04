#!/usr/bin/env python

import json
from math import exp
import os
import random
import pygame
import argparse
import sys
from typing import Union
import uuid

import numpy as np
from numpy.typing import ArrayLike

from skdecide import TransitionOutcome, Value
from skdecide.hub.space.gym import GymSpace, BoxSpace

from ray.rllib.models.modelv2 import flatten, restore_original_dimensions
from ray.rllib.algorithms.ppo import PPOConfig

from generator.configurations.default_configuration import (
    DefaultProblemConfig,
)
from generator.configurations.configs import UnsolvabilityScenario
from beluga_lib.beluga_problem import BelugaProblemDecoder

from skd_domains.skd_base_domain import SkdBaseDomain, PladoState
from skd_domains.skd_pddl_domain import SkdPDDLDomain
from skd_domains.skd_ppddl_domain import SkdPPDDLDomain
from skd_domains.skd_spddl_domain import SkdSPDDLDomain
from skd_domains.skd_gym_domain_tim import BelugaGymCompatibleDomain, BelugaGymEnv
from skdecide import RLDomain, TransitionOutcome, Value, Space
from generate_instance import ProbConfig, main as encode_json
from skd_domains.skd_gym_domain_tim import D
import math

class ExampleBelugaGymCompatibleDomain(BelugaGymCompatibleDomain):
    """This is an example specialization of the BelugaGymCompatibleDomain class
    which transforms PDDL-style states and actions from the original Beluga
    scikit-decide domains to tensors to be used with deep reinforcement learning
    algorithms. As shown at the bottom of this script, an instance of this class
    must be passed to the BelugaGymEnv class, which will automatically populate
    the gymnasium environment methods expected by deep reinforcement learning
    solvers.

    Please note that this class does not provide efficient tensor representations
    for states and actions. It is only intended as exemplifying the transformation
    from PDDL-style states and actions to tensors, and vice-versa.

    State representation.
    The PDDL-style state is composed of a set of predicates - and also fluents in the
    case of the numeric PDDL encoding. Predicate and fluents take the form
    (head obj1 ... objn), e.g. (at-side trailer1 fside). While a predicate can only be
    true or false, a fluent can take a numeric value. For the tensor representation,
    we treat predicates and fluents in the same way, assuming that a predicate is a fluent
    taking Boolean values only. Our tensor representation is a fixed-size tensor,
    but sufficiently large to handle problems of different sizes, using padding to disregard
    the unused tensor entries. The observation space is a matrix NxM where each line
    encodes a fluent and each column encodes the fluent's objects. Since fluents can
    have a different number of arguments, we have as many columns as the the maximum
    number of arguments across all the fluents, and we use padding to disregard entries
    corresponding to unused arguments. It means that a fluent f, whose integer ID is f-id,
    and whose arguments integer IDs are arg1-id, arg2-id, ..., and whose value if f-value,
    is represented as a row vector in the form [f-id arg1-id arg2-id ... argn-id fvalue].
    The arguments which are not used are equal to -1. For instance, if all the fluents
    in the domain use at most 4 arguments, we would encode the Boolean predicate
    (at-side trailer1 fside) taking a true value with a row vector in the form
    [at-side-id trailer1-id f-side-id -1 -1 1].
    IMPORTANT NOTE: in fact, PDDL-style states only enumerate Boolean predicates which
    are true, for memory efficiency reasons. Our tensor representation does the same,
    by listing only the rows corresponding to Boolean predicates that take a true value.
    However, all the integer and float fluents are listed, as in PDDL.

    Action representation.
    The PDDL-style action takes the form of (head obj1 ... objn), e.g.
    (get-from-hangar jig1 hangar1 trailer1). The tensor action is represented as a
    1-dimensional tensor, i.e. a vector, in the form [action-id arg1-id ... argn-id]
    where action-id, arg1-id, ..., argn-id are all integers. The number of argument
    entries is equal to the maximum number of action arguments across all the actions
    in the domain, meaning that unused argument entries for some actions are equal to -1.
    For instance, if the maximum number of action arguments across all the actions in
    the domain is equal to 5, the PDDL action (get-from-hangar jig1 hangar1 trailer1)
    would be encoded as a vector [get-from-hangar-id jig1-ig hangar1-id trailer1-id -1 -1].

    Applicable actions pitfall.
    Whereas only a few actions are application in each possible given state, the number
    of potential actions is huge (exponential in the number of action arguments). This
    prevents from using standard masking techniques in deep reinforcement learning to
    mask inapplicable actions in a given observation, because it would require first to
    build a vector of size equal to the number of potential actions in the problem - which
    is intractable for the Beluga problem. Finding a reasonable way to mask the inapplicable
    actions in the Beluga environment is part of the challenge. In this simplistic
    environment, we assume that all the actions are potentially applicable in each state,
    but we penalize the inapplicable ones (see the reward signal description below). An
    episode systematically ends when an action is applied in the current state where it is
    not applicable.

    Reward signal.
    The objective of the Beluga environment is to find a policy which reaches a goal state
    with a minimum number of steps. In this example environment, we propose to model the
    reward signal as a function between 0 and 1, which is equal to 0 for all steps but the
    terminal step which is equal to:
    - 0 if the goal is not reached in the terminal step (meaning that this terminal step
    corresponds to having applied an inapplicable action in the previous step or to having
    exhausted the step budget)
    - exp(-nb_of_steps) if the terminal step corresponds to a goal situation
    That way, reaching the goal is always better than not reaching it, and it is better to
    reach the goal with the fewest possible number of steps.
    """

    def __init__(
        self,
        skd_beluga_domain: Union[SkdPDDLDomain, SkdPPDDLDomain, SkdSPDDLDomain],
        max_fluent_value: np.int32 = 1000,
        max_nb_atoms_or_fluents: np.int32 = 1000,
        max_nb_steps: np.int32 = 1000,
    ) -> None:
        """The constructor of the example Gym-compatible Beluga domain. For simplicity reasons, we
        construct states as a 3-dimensional tensor (x, y, z) where we store (y, z) as described in
        the summary presentation of the class but for 3 different categories represented as x:
        static facts (x=0), atoms (x=1), fluents (x=2). It is especially useful because the number
        of atoms, whose only the ones which are true are stored in the state, is state-dependent
        contrary to static facts and fluents. Then, we flatten the tensor before transferring it
        to the external world in the reset() and step() methods.

        Args:
            skd_beluga_domain (Union[SkdPDDLDomain, SkdPPDDLDomain, SkdSPDDLDomain]): the original
            Beluga scikit-decide domain, i.e. one of SkdPDDLDomain, SkdPPDDLDomain,or  SkdSPDDLDomain
            max_fluent_value (np.int32, optional): The maximum value that can take a fluent. Defaults to 1000.
            max_nb_atoms_or_fluents (np.int32, optional): The maximum number of static facts, or predicates,
            or fluents. Each of those categories will have max_nb_atoms_or_fluents entries in the tensor. Defaults to 1000.
            max_nb_steps (np.int32, optional): Maximum number of steps per simulation episode. Defaults to 1000.
        """
        super().__init__(skd_beluga_domain)
        self.max_fluent_value: np.int32 = max_fluent_value
        self.max_nb_atoms_or_fluents: np.int32 = max_nb_atoms_or_fluents
        self.max_nb_steps: np.int32 = max_nb_steps
        
        #TODO find way of gating calcling this stuff behind a render mode
        #TOD maybe its just faster to reconvert to json or pass the json in than calcuing all this

        #TODO there is already mappings in skd_beluga_domain._action_idx, skd_beluga_domain._predicate_idx etc by __init__deserializer()
        self.pred_mapping = {p.name: i for i, p in enumerate(self.skd_beluga_domain.task.predicates) 
                           if i < self.skd_beluga_domain.task.num_fluent_predicates}
        self.pred_mapping_inv = {v: k for k, v in self.pred_mapping.items()}
        self.obj_mapping = {i: o for i, o in enumerate(self.skd_beluga_domain.task.objects)}
        self.obj_mapping_inv = {o: i for i, o in self.obj_mapping.items()}
        self.racks ={v: r  for r,v in self.obj_mapping.items() if "rack" in v}
        self.trailers = {v: r  for r,v in self.obj_mapping.items() if "trailer" in v}
        self.hangars = {v: r  for r,v in self.obj_mapping.items() if "hangar" in v}
        self.static_facts_mapping = {p.name: i for i, p in enumerate(self.skd_beluga_domain.task.predicates) 
                           if i >= self.skd_beluga_domain.task.num_fluent_predicates}
        self.domain_action_idx = skd_beluga_domain._action_idx
        self.domain_action_ind_inv = {v: k for k, v in self.domain_action_idx.items()}
        self.inital_flight_obj_num = tuple(self.skd_beluga_domain.task.initial_state.atoms[self.pred_mapping["processed-flight"]])[0][0]
        self.flight_list = self.get_flight_list(self.inital_flight_obj_num) #TODO modify as changes thru episode
        self.inital_unloads = self.skd_beluga_domain.task.initial_state.atoms[self.pred_mapping['to_unload']]
        
        self.initial_loads = self.skd_beluga_domain.task.initial_state.atoms[self.pred_mapping['to_load']]
        
        self.inital_deliveries = self.skd_beluga_domain.task.initial_state.atoms[self.pred_mapping['to_deliver']]
        
       


#Rendering

        self.window_size_x = 1200
        self.window_size_y = 1000        
        self.screen = None
        self.fps = 60
        self.clock = pygame.time.Clock()
        
        #Spaces
        self.true_observation_space = BoxSpace(
            low=np.ones(
                shape=[
                    3,
                    self.max_nb_atoms_or_fluents,
                    2
                    + max(
                        len(p.parameters)
                        for p in self.skd_beluga_domain.task.predicates
                    ),
                ],
                dtype=np.int32,
            )
            * (-1),
            high=np.array(
                [
                    [
                        [
                            len(self.skd_beluga_domain.task.predicates),
                        ]
                        + (
                            [len(self.skd_beluga_domain.task.objects)]
                            * max(
                                len(p.parameters)
                                for p in self.skd_beluga_domain.task.predicates
                            )
                        )
                        + [
                            (
                                1
                                if len(self.skd_beluga_domain.task.functions) == 0
                                else self.max_fluent_value
                            )
                        ]
                    ]
                    * self.max_nb_atoms_or_fluents
                ]
                * 3,
                dtype=np.int32,
            ),
        )
        self.observation_space = BoxSpace(
            self.true_observation_space.unwrapped().low.flatten(),
            self.true_observation_space.unwrapped().high.flatten(),
            dtype=self.true_observation_space.unwrapped().dtype,
        )
        self.action_space = BoxSpace(
            low=np.array(
                [0]
                + (
                    [-1]
                    * max(p.parameters for p in self.skd_beluga_domain.task.actions)
                ),
                dtype=np.int32,
            ),
            high=np.array(
                [
                    len(self.skd_beluga_domain.task.actions),
                ]
                + (
                    [len(self.skd_beluga_domain.task.objects)]
                    * max(a.parameters for a in self.skd_beluga_domain.task.actions)
                ),
                dtype=np.int32,
            ),
        )
        self.current_pddl_state: SkdBaseDomain.T_state = None
        self.nb_steps: int = 0
    
    #Methods
    @property
    def unload_list(self):
        return self.get_unload_list(self.inital_unloads) #TODO modify as changes thru episode
    @property
    def load_list(self):
        return self.get_load_list(self.initial_loads) #TODO modify as changes thru episode
    @property
    def delivery_list(self):
        return self.get_del_list(self.inital_deliveries) #TODO modify as changes thru episode
    
    def get_load_list(self, initial_loads):
        load_list = {}
        next_load_set = {i for i in self.skd_beluga_domain.task.static_facts[self.static_facts_mapping['next_load']-len(self.pred_mapping)]}
        for load in initial_loads: #(type, in slot, plane)
            load_list[self.obj_mapping[load[2]]] = [[self.obj_mapping[load[0]], self.obj_mapping[load[1]]]]
            next_slot = self.obj_mapping[load[1]]#next slot to load
            while not next_slot == "dummy-slot":
                for next_load_quad in next_load_set:
                    if self.obj_mapping[next_load_quad[1]] == next_slot:
                        if not [self.obj_mapping[next_load_quad[0]], self.obj_mapping[next_load_quad[1]]] in load_list[self.obj_mapping[load[2]]]:
                            load_list[self.obj_mapping[load[2]]].append([self.obj_mapping[next_load_quad[0]], self.obj_mapping[next_load_quad[1]]])
                        next_slot = self.obj_mapping[next_load_quad[2]]
                        next_load_set.remove(next_load_quad)
                        break
        current_loads= {self.obj_mapping[k]:[self.obj_mapping[i], self.obj_mapping[j]] for (i,j,k) in self.current_pddl_state.atoms[self.pred_mapping['to_load']]}
        for k,v in current_loads.items():
            load_list[k] = load_list[k][load_list[k].index(v):]    #TODO not checked if this is correct            
        return load_list
    
    def get_del_list(self, initial_unloads):
        unload_list = {}
        next_unload_set = {i for i in self.skd_beluga_domain.task.static_facts[self.static_facts_mapping['next_deliver']-len(self.pred_mapping)]}
        for unload in initial_unloads:
            unload_list[self.obj_mapping[unload[1]]] = [self.obj_mapping[unload[0]]]
            next_unload = self.obj_mapping[unload[0]]
            while not next_unload == 'dummy-jig':
                for next_unload_pair in next_unload_set:
                    if self.obj_mapping[next_unload_pair[0]] == next_unload:
                        unload_list[self.obj_mapping[unload[1]]].append(self.obj_mapping[next_unload_pair[1]])
                        next_unload = self.obj_mapping[next_unload_pair[1]]
                        next_unload_set.remove(next_unload_pair)
                        break


        current_empty= [self.obj_mapping[i[0]] for i in self.current_pddl_state.atoms[self.pred_mapping['empty']]]
        for k,v in unload_list.items():
            unload_list[k] = [i for i in v if not i in current_empty] #TODO not checked if this is correct
        return unload_list 
    
    def get_unload_list(self, initial_unloads):
        unload_list = {}
        next_unload_set = {i for i in self.skd_beluga_domain.task.static_facts[self.static_facts_mapping['next_unload']-len(self.pred_mapping)]}
        for unload in initial_unloads:
            unload_list[self.obj_mapping[unload[1]]] = [self.obj_mapping[unload[0]]]
            next_unload = self.obj_mapping[unload[0]]
            while not next_unload == 'dummy-jig':
                for next_unload_pair in next_unload_set:
                    if self.obj_mapping[next_unload_pair[0]] == next_unload:
                        unload_list[self.obj_mapping[unload[1]]].append(self.obj_mapping[next_unload_pair[1]])
                        next_unload = self.obj_mapping[next_unload_pair[1]]
                        next_unload_set.remove(next_unload_pair)
                        break

        current_unloads= {self.obj_mapping[j]:self.obj_mapping[i] for (i,j) in self.current_pddl_state.atoms[self.pred_mapping['to_unload']]}
        for k,v in current_unloads.items():
            unload_list[k] = unload_list[k][unload_list[k].index(v):]
        return unload_list
    
    def jig_empty(self, jig: str):
        if not jig:
            return False
        num = self.obj_mapping_inv[jig]
        empty_list = [i[0] for i in self.current_pddl_state.atoms[self.pred_mapping['empty']]]
        if num in empty_list:
            return True
        else:
            return False
        



            







    def get_flight_list(self, initial_flight):
        flight_list = [initial_flight]
        flight_pairs = {f for f in self.skd_beluga_domain.task.static_facts[self.static_facts_mapping['next-flight-to-process']-len(self.pred_mapping)]}
        while flight_pairs:
            for pair in flight_pairs:
                if pair[0] == initial_flight:
                    flight_list.append(pair[1])
                    flight_pairs.remove(pair)
                    initial_flight = pair[1]
                    break
        return [self.obj_mapping[f] for f in flight_list]


    def pddl_state_to_english(self, pddl_state: SkdBaseDomain.T_state = None) -> str:
        if not pddl_state:
            pddl_state = self.current_pddl_state
        atom_dict = {}
        for idx, atoms in enumerate(pddl_state.atoms):
            atom_list = []
            header = self.pred_mapping_inv[idx]
            for atom in atoms:
                a = tuple(self.obj_mapping[i] for i in atom)
                atom_list.append(a)
            atom_dict[header] = atom_list
        return atom_dict
        
    def action_to_english(self, action_args: tuple) -> str:
        arg_names = [self.obj_mapping[i] for i in action_args[1:] if i in self.obj_mapping]
        act = self.domain_action_ind_inv[action_args[0]]
        return f"{act} {', '.join(arg_names)}"


    def make_state_array(self, pddl_state: SkdBaseDomain.T_state) -> ArrayLike:
        state_array: ArrayLike = np.ones(
            shape=self.true_observation_space.shape, dtype=np.int32
        ) * (-1)
        i = 0
        for p, atom in enumerate(self.skd_beluga_domain.task.static_facts):
            for args in atom:
                if i >= self.max_nb_atoms_or_fluents:
                    raise RuntimeError(
                        "Too many static atoms to store them in the state tensor; "
                        "please increase max_nb_atoms_or_fluents"
                    )
                state_array[0][i][0] = p
                state_array[0][i][1 : 1 + len(args)] = args
                state_array[0][i][-1] = 1
                i += 1
        i = 0
        for p, atom in enumerate(pddl_state.atoms):
            for args in atom:
                if i >= self.max_nb_atoms_or_fluents:
                    raise RuntimeError(
                        "Too many state atoms to store them in the state tensor; "
                        "please increase max_nb_atoms_or_fluents"
                    )
                state_array[1][i][0] = p
                state_array[1][i][1 : 1 + len(args)] = args
                state_array[1][i][-1] = 1
                i += 1
        i = 0
        for f, fluent in enumerate(pddl_state.fluents):
            for args in fluent:
                if i >= self.max_nb_atoms_or_fluents:
                    raise RuntimeError(
                        "Too many state fluents to store them in the state tensor; "
                        "please increase max_nb_atoms_or_fluents"
                    )
                state_array[2][i][0] = f
                state_array[2][i][1 : 1 + len(args)] = args
                state_array[2][i][-1] = fluent[args]
                i += 1
        return state_array.flatten()

    def _get_applicable_actions_from(self, pddl_state) -> Space[D.T_event]:
        return self.skd_beluga_domain._get_applicable_actions_from(
            pddl_state
        )
    
    def make_pddl_state(self, state_array: ArrayLike) -> SkdBaseDomain.T_state:
        plado_state = PladoState(
            num_predicates=len(self.skd_beluga_domain.task.predicates),
            num_functions=len(self.skd_beluga_domain.task.functions),
        )
        restored_state_array: ArrayLike = state_array.reshape(
            self.true_observation_space.shape
        )
        for atom in range(self.max_nb_atoms_or_fluents):
            if (
                restored_state_array[1][atom][0] >= 0
                and restored_state_array[1][atom][-1] >= 0
            ):
                plado_state.atoms[restored_state_array[1][atom][0]].add(tuple(
                    
                        int(arg)
                        for arg in restored_state_array[1][atom][1:-1]
                        if arg >= 0
                )
                )
        for fluent in range(self.max_nb_atoms_or_fluents):
            if restored_state_array[2][fluent][0] >= 0:
                plado_state.fluents[restored_state_array[2][fluent][0]].update(
                    {
                        [
                            int(arg)
                            for arg in restored_state_array[2][fluent][1:-1]
                            if arg >= 0
                        ]: restored_state_array[2][fluent][-1]
                    }
                )
        return SkdBaseDomain.T_state(
            domain=self.skd_beluga_domain,
            state=plado_state,
            cost_function=self.skd_beluga_domain.cost_functions,
        )

    def make_action_array(self, pddl_action: SkdBaseDomain.T_event) -> ArrayLike:
        action_array: ArrayLike = np.ones(
            shape=self.action_space.shape, dtype=np.int32
        ) * (-1)
        action_array[0] = pddl_action.action_id
        action_array[1 : 1 + len(pddl_action.args)] = pddl_action.args
        return action_array

    def make_pddl_action(self, action_array: ArrayLike) -> SkdBaseDomain.T_event:
        return SkdBaseDomain.T_event(
            domain=self.skd_beluga_domain,
            action_id=int(action_array[0]),
            args=tuple(int(arg) for arg in action_array[1:] if arg >= 0),
        )
    def render(self, mode='human'):
        self._render_frame()

    def _state_reset(self) -> BelugaGymCompatibleDomain.T_state:
        self.nb_steps = 0
        self.current_pddl_state = self.skd_beluga_domain._state_reset()
        return self.make_state_array(self.current_pddl_state)

    def _state_step(
        self, action: BelugaGymCompatibleDomain.T_event
    ) -> TransitionOutcome[
        BelugaGymCompatibleDomain.T_state,
        Value[BelugaGymCompatibleDomain.T_value],
        BelugaGymCompatibleDomain.T_predicate,
        BelugaGymCompatibleDomain.T_info,
    ]:
        """when step is called it cascasdes through various skdcide fucntions and this func is finally called by
        
        /opt/anaconda3/envs/Beluga_latest/lib/python3.11/site-packages/skdecide/builders/domain/dynamics.py

        NOTE that calling step then gives the outcome in the above file to which this func is just an input the return of the step func is 

        return EnvironmentOutcome(
            observation,
            transition_outcome.value,
            transition_outcome.termination,
            transition_outcome.info,
        )
        """
        #events = pygame.event.get() #TODO Needs to have had pygame initialed so this needs ot be in inital or reset and only relevant on render
        self.nb_steps += 1
        pddl_action: SkdBaseDomain.T_event = self.make_pddl_action(action)

        if self.skd_beluga_domain._get_applicable_actions_from(
            self.current_pddl_state
        ).contains(pddl_action):
            outcome = self.skd_beluga_domain._state_step(pddl_action)
            self.current_pddl_state = outcome.state
            array_state = self.make_state_array(outcome.state)
            return TransitionOutcome(
                state=array_state,
                value=Value(reward=exp(-self.nb_steps) if outcome.termination else 0),
                termination=outcome.termination or self.nb_steps >= self.max_nb_steps,
                info=outcome.info,
            )
        else:
            return TransitionOutcome(
                state=self.make_state_array(self.current_pddl_state),
                value=Value(reward=0),
                termination=True,
                info=None,
            )

    def _get_observation_space_(
        self,
    ) -> GymSpace[BelugaGymCompatibleDomain.T_observation]:
        return self.observation_space

    def _get_action_space_(self) -> GymSpace[BelugaGymCompatibleDomain.T_event]:
        return self.action_space

    

    ##########################################################RENDERING#########################################################
    
    def _draw_racks(self):
        num_racks = len(self.racks)
        self.rows = math.ceil(num_racks / 10) 
        self.racks_per_row = math.ceil(num_racks / self.rows ) 
        self.rack_draw_width = (self.window_size_x - 200 - ((self.racks_per_row-1)*10)) / self.racks_per_row
        self.rack_draw_height = 200/self.rows
        row = 0
        row_counter = 0
        for i, rack in enumerate(self.racks.keys()):

            
            pygame.draw.rect(self.screen, (0, 0, 0), (100+ row_counter* self.rack_draw_width + 10*row_counter, #x_start
                                                      self.window_size_y/3 + row * self.rack_draw_height + row*20, #y_start
                                                      self.rack_draw_width, #width
                                                      self.rack_draw_height)) #height
            row_counter += 1
            if row_counter >= self.racks_per_row:
                row += 1
                row_counter = 0

    def draw_rack_jigs(self):
        self.jigs = {self.obj_mapping[i]:self.obj_mapping[j] for (i,j) in self.current_pddl_state.atoms[self.pred_mapping["in"]]}
        self.font = pygame.font.SysFont("Arial", 10)
        row = 0
        row_counter = 0
        free_spaces = {k:v for (k,v) in self.current_pddl_state.atoms[self.pred_mapping["free-space"]]}
        curr_jig_sizes = {self.obj_mapping[i]:int(self.obj_mapping[j][1:]) for (i,j) in self.current_pddl_state.atoms[self.pred_mapping["size"]]}
        for i, rack in enumerate(self.racks.keys()):
            
            jigs = [k for k,v in self.jigs.items() if v == rack]
            
            empty_space = free_spaces[self.obj_mapping_inv[rack] ]
            for j, jig in enumerate(jigs):
                jig_size = curr_jig_sizes[jig]
                if self.jig_empty(jig):
                    colour=(255,0,0) #red
                else:
                    colour=(0,255,0) #green
                text_surface = self.font.render(jig+ ' '+f"({jig_size})", False, colour)
                text_loc = (100+ row_counter* self.rack_draw_width + 10*row_counter +self.rack_draw_width/2-21,
                            (self.window_size_y/3) + self.rack_draw_height/2 + len(jigs)*10/2 - 12 - j*10 +row*self.rack_draw_height + 20*row) #co-ords starts top left x, y
                self.screen.blit(text_surface,text_loc)

            
            
            text_surf_rack_size = self.font.render(f"Empty_space: {empty_space}", False, (0,0,0))
            text_loc_rack_size = (100+ row_counter* self.rack_draw_width + 10*row_counter +(self.rack_draw_width/2)-33,
                                   (self.window_size_y/3) -12 + row * self.rack_draw_height +row*20)
            self.screen.blit(text_surf_rack_size,text_loc_rack_size)
            row_counter += 1
            if row_counter >= self.racks_per_row:
                row += 1
                row_counter = 0

    def draw_trailers(self):
        factory_trailers = [i for i in self.trailers.keys() if "factory" in i]
        beluga_trailers =  [i for i in self.trailers.keys() if "beluga" in i]
        num_f = len(factory_trailers)
        num_b = len(beluga_trailers)
        f_draw_width = ((self.window_size_x - 200 - ((num_f-1) *10))/num_f)
        b_draw_width = ((self.window_size_x - 200 - ((num_b-1) *10))/num_b)
        t_draw_height = 30
        for i, trailer in enumerate(beluga_trailers):
            contents = [self.obj_mapping[item] for(item, loc) in self.current_pddl_state.atoms[3] if loc == self.trailers[trailer]]
            colour = (100,0,0)
            pygame.draw.rect(self.screen, colour, ( 100 + i*b_draw_width + 10*i,
                                                   self.window_size_y - 100,
                                                   b_draw_width,
                                                   t_draw_height
                                                   ))
           
            text_surface_trailer_conts = self.font.render(f"{contents}", False, (0,0,0))
            text_loc_trailer_conts = (100 + i*b_draw_width + 10*i + b_draw_width/2,self.window_size_y - 100 + t_draw_height/2 - 12)
            self.screen.blit(text_surface_trailer_conts,text_loc_trailer_conts)
        for i, trailer in enumerate(factory_trailers):
            contents = [self.obj_mapping[item] for(item, loc) in self.current_pddl_state.atoms[3] if loc == self.trailers[trailer]]
            colour = (0,255,0)
            pygame.draw.rect(self.screen, colour, ( 100 + i*f_draw_width + 10*i,
                                                   100,
                                                   f_draw_width,
                                                   t_draw_height
                                                   ))
            text_surface_trailer_conts = self.font.render(f"{contents}", False, (0,0,0))
            text_loc_trailer_conts = (100 + i*f_draw_width + 10*i + f_draw_width/2,100 + t_draw_height/2 - 12)
            self.screen.blit(text_surface_trailer_conts,text_loc_trailer_conts)


    def render_unloads_loads(self):

        unload = [(i, self.unload_list[i]) for i in self.flight_list]
        text_surface_unloads = self.font.render(f"{unload}", False, (0,0,0))
        text_loc_unloads = (50, self.window_size_y - 20)
        self.screen.blit(text_surface_unloads,text_loc_unloads)

        text_surface_flight = self.font.render(f"Flights:{self.flight_list}", False, (0,0,0))
        text_loc_flight = (50, self.window_size_y - 50)
        self.screen.blit(text_surface_flight,text_loc_flight)

        text_suface_loads = self.font.render(f"Loads:{self.load_list}", False, (0,0,0))
        text_loc_loads = (50, self.window_size_y - 65)
        self.screen.blit(text_suface_loads,text_loc_loads)

        text_suface_deliveries = self.font.render(f"Deliveries:{self.delivery_list}", False, (0,0,0))
        text_loc_deliveries = (50, 10)
        self.screen.blit(text_suface_deliveries,text_loc_deliveries)

        






            
           



                


    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((self.window_size_x, self.window_size_y))  # Adjust size as needed
            pygame.display.set_caption("Beluga Board")
       
        """running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False"""
            
        self.screen.fill((255, 255, 255))  # White background

        self._draw_racks()
        self.draw_rack_jigs()
        self.draw_trailers()
        self.render_unloads_loads()
        pygame.display.update()
        self.clock.tick(0.5) #Ticks per second
        




           
        
        

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


    ###################################################################################
    

if __name__ == "__main__":
    inst = "/Users/ha24583/Documents/GitHub/Beluga-AI-Challenge/instances/toy"
    problem_name = "toy"
    problem_folder = os.path.join("/Users/ha24583/Documents/GitHub/Beluga-AI-Challenge-internal/tools", "output") #TODO for some reason this needs to be a path sigh
    probabilistic = False
    probabilistic_model = "arrivals" #NB this isnt used if probabilistic is false



    with open(inst, "r") as fp:
        inst = json.load(fp, cls=BelugaProblemDecoder) #This overwrites the json decode method (by subclassing json decode to decode as a Beluga problem
    domain_factory = lambda: (
        SkdPPDDLDomain(inst, problem_name, problem_folder)
        if probabilistic and probabilistic_model == "ppddl"
        else (
            SkdSPDDLDomain(inst, problem_name, problem_folder)
            if probabilistic and probabilistic_model == "arrivals"
            else SkdPDDLDomain(inst, problem_name, problem_folder)
        )
    )
    domain = domain_factory()

    env_config={"domain": ExampleBelugaGymCompatibleDomain(domain)}
    env=BelugaGymEnv(env_config)
    #BelguaGym wraps AsGymnasiumEnv which passes the "domain" to a gymnasium style constructor. The AsLegacyGymV21Env has the following functions which are called when you do env.{func}
    #step, reset, render, close, seed
    #it also puts the domain in a _domain attribute so to refer to functions specifcalluy in ExampleBelugaGymCompatibleEnv you need to use env.env._domain.{func} UNLESS its one of the above in which
    #you can use env.{func}
    #specifically
    #env.step(action) calls EnvCompatibility step which calls AsLegacyGymV21Env step (which has some code for gettings actions spaces etc so might be casue of isseus somewhen)
    #but calls ._domain.step which becuase EXAMPLEBELGUA doenst have one it calls the base class one which is in the dynamics.py file
    #this calls the _state_step function in ExampleBelugaGymCompatibleDomain
    #this is complicated so probably dont mess with it

    #TODO
    obs,_ = env.reset()
    env.render()
    #a_s = env.action_space.sample()
    act = [1,38,39,32,50,-1,-1,-1,-1,-1]
    act_2 = [4,38,32,36,0,23,29,13,-1,-1]
    #this is passed as array which is processed in the _state_step func to make it a PDDL action
    eng = env.env._domain.pddl_state_to_english()
    transition_outcome = env.step(act)
    env.render()
    
    transition_outcome = env.step(act_2)
    #needs to pass D-Tstate to gaa
    env.render() #passes through various buyt effectively calls ExampleBelugaGymCompatibleDomain.render
    aa = env.env._domain._get_applicable_actions_from(env.env._domain.current_pddl_state)
    print("stop")

    

    """
    array to PDDL action conversion

    def make_pddl_action(self, action_array: ArrayLike) -> SkdBaseDomain.T_event:
    return SkdBaseDomain.T_event(
        domain=self.skd_beluga_domain,  
        action_id=int(action_array[0]), #first digit is action id cast to INT (this rounds down?)
        args=[int(arg) for arg in action_array[1:] if arg >= 0], #then each arg that is above 0
    )
    """

    #domain._action_idx = {'load-beluga': 0, 'unload-beluga': 1, 'get-from-hangar': 2, 'deliver-to-hangar': 3, 
    # 'put-down-rack': 4, 'stack-rack': 5, 'pick-up-rack': 6, 'unstack-rack': 7, 'beluga-complete': 8}

    """
    0 =
<skd_domains.skd_base_domain.Action object at 0x32b5fb550>
special variables
action_id =
1
args =
(38, 39, 32, 50)
domain =
<skd_domains.skd_pddl_domain.SkdPDDLDomain object at 0x31849c510>
1 =
<skd_domains.skd_base_domain.Action object at 0x32b5fac10>
special variables
action_id =
1
args =
(38, 39, 33, 50)
domain =
<skd_domains.skd_pddl_domain.SkdPDDLDomain object at 0x31849c510>
len() =
2
_gym_space =
Discrete(2)
"""