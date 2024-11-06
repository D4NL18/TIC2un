import { TestBed } from '@angular/core/testing';

import { DeeplearningService } from './deeplearning.service';

describe('DeeplearningService', () => {
  let service: DeeplearningService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(DeeplearningService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
