"""CPU checks for complete context batches and all FlowWrapper entry points.

Run: python -m unittest discover -s External/dingo/tests/core -p test_flow_wrapper_context.py -v
"""

import unittest

import torch
from dingo.core.nn.enets import ModuleMerger
from dingo.core.nn.nsf import (
    FlowWrapper, create_nsf_model, create_nsf_with_rb_projection_embedding_net,
)


class TestFlowWrapperContext(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def flow_kwargs(self, context_dim=5, input_dim=4):
        return dict(input_dim=input_dim, context_dim=context_dim, num_flow_steps=2,
                    base_transform_kwargs=dict(hidden_dim=12, num_transform_blocks=1,
                        activation='elu', dropout_probability=0.0, norm='BatchNorm',
                        num_bins=4, base_transform_type='rq-coupling'))

    def check_sampling(self, model, args, embedded, batch):
        shape = (3, 4) if batch is None else (batch, 3, 4)
        torch.manual_seed(71)
        actual = model.sample(*args, num_samples=3)
        torch.manual_seed(71)
        expected = model.flow.sample(3, context=embedded)
        self.assertEqual(tuple(actual.shape), shape)
        torch.testing.assert_close(actual, expected)
        torch.manual_seed(72)
        samples, logp = model.sample_and_log_prob(*args, num_samples=3)
        torch.manual_seed(72)
        expected_samples, expected_logp = model.flow.sample_and_log_prob(3, context=embedded)
        self.assertEqual(tuple(samples.shape), shape)
        self.assertEqual(tuple(logp.shape), shape[:-1])
        torch.testing.assert_close(samples, expected_samples)
        torch.testing.assert_close(logp, expected_logp)

    def test_embedding_preserves_singleton_and_larger_batches(self):
        for batch in (1, 3):
            with self.subTest(batch=batch):
                embedding = torch.nn.Sequential(torch.nn.Linear(3, 5))
                model = FlowWrapper(create_nsf_model(**self.flow_kwargs()), embedding).eval()
                context = torch.randn(batch, 3, requires_grad=True)
                theta = torch.randn(batch, 4)
                embedded = embedding(context)
                actual = model(theta, context)
                self.assertEqual(tuple(actual.shape), (batch,))
                torch.testing.assert_close(actual, model.flow.log_prob(theta, context=embedded))
                (-actual.mean()).backward()
                self.assertIsNotNone(embedding[0].weight.grad)
                self.assertIsNotNone(context.grad)
                self.assertTrue(torch.isfinite(context.grad).all())
                self.check_sampling(model, (context,), embedded, batch)

    def test_raw_context_tensor_is_passed_intact(self):
        model = FlowWrapper(create_nsf_model(**self.flow_kwargs())).eval()
        context = torch.randn(3, 5)
        theta = torch.randn(3, 4)
        for args in ((context,), ((context,),), ([context],)):
            torch.testing.assert_close(model.log_prob(theta, *args),
                                       model.flow.log_prob(theta, context=context))
            self.check_sampling(model, args, context, 3)

    def test_no_context_and_explicit_none(self):
        model = FlowWrapper(create_nsf_model(**self.flow_kwargs(context_dim=None))).eval()
        theta = torch.randn(3, 4)
        expected = model.flow.log_prob(theta)
        for args in ((), (None,), ((),), ([],)):
            with self.subTest(args=args):
                torch.testing.assert_close(model(theta, *args), expected)
                self.check_sampling(model, args, None, None)

    def test_multiple_embedding_inputs_and_legacy_packed_arguments(self):
        embedding = ModuleMerger((torch.nn.Linear(3, 3), torch.nn.Identity()))
        model = FlowWrapper(create_nsf_model(**self.flow_kwargs()), embedding).eval()
        strain, proxy = torch.randn(3, 3), torch.randn(3, 2)
        theta = torch.randn(3, 4)
        embedded = embedding(strain, proxy)
        expected = model.flow.log_prob(theta, context=embedded)
        for args in ((strain, proxy), ((strain, proxy),), ([strain, proxy],)):
            torch.testing.assert_close(model.log_prob(theta, *args), expected)
            self.check_sampling(model, args, embedded, 3)

    def test_raw_flow_rejects_multiple_context_arguments(self):
        model = FlowWrapper(create_nsf_model(**self.flow_kwargs())).eval()
        context = torch.randn(3, 5)
        with self.assertRaisesRegex(ValueError, 'expects one context tensor'):
            model.log_prob(torch.randn(3, 4), context, context)
        with self.assertRaisesRegex(ValueError, 'expects one context tensor'):
            model.sample(context, context)
        with self.assertRaisesRegex(ValueError, 'expects one context tensor'):
            model.sample_and_log_prob(context, context)

    def test_full_30d_model_builder_forward_and_backward(self):
        model = create_nsf_with_rb_projection_embedding_net(
            self.flow_kwargs(input_dim=30),
            dict(input_dims=(2, 3, 8), V_rb_list=None, svd={'size': 2},
                 output_dim=5, hidden_dims=[12, 8], activation='elu',
                 dropout=0.0, norm='BatchNorm', added_context=False),
        ).eval()
        theta = torch.randn(3, 30)
        strain = torch.randn(3, 2, 3, 8)
        density = model(theta, strain)
        self.assertEqual(tuple(density.shape), (3,))
        self.assertTrue(torch.isfinite(density).all())
        (-density.mean()).backward()
        projection = model.embedding_net[0].layers_rb[0]
        self.assertIsNotNone(projection.weight.grad)
        self.assertTrue(torch.isfinite(projection.weight.grad).all())
        samples, logp = model.sample_and_log_prob(strain, num_samples=2)
        self.assertEqual(tuple(samples.shape), (3, 2, 30))
        self.assertEqual(tuple(logp.shape), (3, 2))


if __name__ == '__main__':
    unittest.main()
