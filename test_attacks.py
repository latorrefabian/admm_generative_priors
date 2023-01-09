# import pdb
from src.util import attack_2
from src.models.loader import load_classifier
from src.experiments import load_test_images

ckpt_mnist = 'pretrained_classifier/mnist_0_1/300-300-300-0.0.ckpt'


def main():
    x, y = load_test_images(
            'mnist', 100, image_type='real', g=None, return_y=True)
    g = load_classifier(model2load=ckpt_mnist, dataset='mnist')
    test_error = attack_2.test_error(g)
    perturb = attack_2.pgm(
            classifier=g, n_iters=30, epsilon=.2, p=float('inf'))
    x_adv = perturb(x, y)
    clean_error = test_error(x, y)
    adv_error = test_error(x_adv, y)
    print('clean error: ' + str(clean_error) + '%')
    print('adversarial error: ' + str(adv_error) + '%')


if __name__ == '__main__':
    main()

